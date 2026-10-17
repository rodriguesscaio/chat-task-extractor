import json
import os
from datetime import date, datetime
from typing import Literal, Optional

import anthropic
import stripe
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

MAX_CHARS = 30_000
FREE_CREDITS = int(os.environ.get("FREE_CREDITS", "3"))
PACK_CREDITS = 10
PACK_PRICE_CENTS = 500
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")

SYSTEM_PROMPT = """You are an assistant that reads chat conversations and extracts actionable tasks.

You will receive a chat log (for example, a WhatsApp export or plain text messages).
Read the whole conversation and return ONLY valid JSON, with no text before or after it.

What counts as a task:
- Someone agrees to do something ("I'll send the report", "ok, I can call him")
- Someone is asked to do something and it is not refused
- A decision that requires an action ("we need to book the room")

What is NOT a task:
- Jokes, greetings, opinions, general discussion
- Things already completed in the chat, unless they are confirmed done (then use status "done")

Rules:
- owner: the person's name as it appears in the chat. Use null if unclear.
- due_date: ISO format YYYY-MM-DD only if a date or deadline is stated or clearly implied
  ("tomorrow", "Friday") relative to the message date. Otherwise null.
- priority: "high" if urgent words or a close deadline, "low" if optional, otherwise "medium".
- status: "done" if the chat confirms it was completed, otherwise "open".
- source_quote: a short quote (max 20 words) from the message that created the task.
- Do not invent tasks, people or dates. If there are no tasks, return an empty list.
- Merge duplicates: if the same task is mentioned several times, return it once.
- Write titles in the same language as the chat, starting with a verb ("Send the budget to Ana").

Return this JSON structure:
{
  "summary": "2-3 sentences about what is happening in the conversation",
  "topics": ["short topic", "short topic"],
  "tasks": [
    {
      "title": "string",
      "description": "string or null",
      "owner": "string or null",
      "due_date": "YYYY-MM-DD or null",
      "priority": "high | medium | low",
      "status": "open | done",
      "source_quote": "string"
    }
  ]
}"""

# ---------- Database ----------

engine = create_engine("sqlite:///tasks.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class Chat(Base):
    __tablename__ = "chats"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    raw_text: Mapped[str] = mapped_column(Text)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    topics: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    tasks: Mapped[list["Task"]] = relationship(back_populates="chat", cascade="all, delete-orphan")


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id"))
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    owner: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    due_date: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    priority: Mapped[str] = mapped_column(String(10), default="medium")
    status: Mapped[str] = mapped_column(String(10), default="open")
    source_quote: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    chat: Mapped[Chat] = relationship(back_populates="tasks")


class Wallet(Base):
    __tablename__ = "wallets"
    client_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    credits: Mapped[int] = mapped_column(Integer, default=0)


class Payment(Base):
    __tablename__ = "payments"
    session_id: Mapped[str] = mapped_column(String(255), primary_key=True)  # Stripe Checkout Session id
    client_id: Mapped[str] = mapped_column(String(64))
    credits: Mapped[int] = mapped_column(Integer)
    amount_cents: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- LLM output schema ----------

class ExtractedTask(BaseModel):
    title: str
    description: Optional[str] = None
    owner: Optional[str] = None
    due_date: Optional[date] = None
    priority: Literal["high", "medium", "low"] = "medium"
    status: Literal["open", "done"] = "open"
    source_quote: str = ""


class Extraction(BaseModel):
    summary: str
    topics: list[str] = Field(default_factory=list)
    tasks: list[ExtractedTask] = Field(default_factory=list)


class StatusUpdate(BaseModel):
    status: Literal["open", "done"]


# ---------- Extraction ----------

def call_llm(chat_text: str) -> str:
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise HTTPException(500, "LLM_MODEL environment variable is not set")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": chat_text}],
        )
    except anthropic.APIError as e:
        raise HTTPException(502, f"LLM request failed: {e}")
    return "".join(b.text for b in resp.content if b.type == "text")


def parse_extraction(raw: str) -> Extraction:
    text = raw.strip()
    # Tolerate a markdown code fence around the JSON
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return Extraction.model_validate(json.loads(text))


def extract(chat_text: str) -> Extraction:
    last_error = None
    for _ in range(2):  # first attempt + one retry
        try:
            return parse_extraction(call_llm(chat_text))
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
    raise HTTPException(502, f"LLM returned invalid JSON after 2 attempts: {last_error}")


# ---------- API ----------

app = FastAPI(title="Chat Task Extractor")


def get_wallet(db: Session, client_id: str) -> Wallet:
    if not client_id or len(client_id) > 64:
        raise HTTPException(400, "Missing or invalid client id")
    wallet = db.get(Wallet, client_id)
    if not wallet:
        wallet = Wallet(client_id=client_id, credits=FREE_CREDITS)
        db.add(wallet)
        db.commit()
    return wallet


def require_stripe():
    if not stripe.api_key:
        raise HTTPException(500, "STRIPE_SECRET_KEY environment variable is not set")


class CheckoutRequest(BaseModel):
    client_id: str


def task_to_dict(t: Task) -> dict:
    return {
        "id": t.id,
        "chat_id": t.chat_id,
        "title": t.title,
        "description": t.description,
        "owner": t.owner,
        "due_date": t.due_date,
        "priority": t.priority,
        "status": t.status,
        "source_quote": t.source_quote,
        "created_at": t.created_at.isoformat(),
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.post("/api/chats")
async def upload_chat(
    file: UploadFile = File(...),
    x_client_id: str = Header(""),
    db: Session = Depends(get_db),
):
    wallet = get_wallet(db, x_client_id)
    if wallet.credits < 1:
        raise HTTPException(402, "No credits left. Buy more to keep extracting.")
    if not (file.filename or "").lower().endswith(".txt"):
        raise HTTPException(400, "Please upload a .txt file")
    raw = (await file.read()).decode("utf-8", errors="replace")
    if not raw.strip():
        raise HTTPException(400, "The file is empty")

    chat = Chat(filename=file.filename, raw_text=raw)
    db.add(chat)
    db.commit()

    result = extract(raw[-MAX_CHARS:])

    chat.summary = result.summary
    chat.topics = json.dumps(result.topics)
    for t in result.tasks:
        db.add(Task(
            chat_id=chat.id,
            title=t.title,
            description=t.description,
            owner=t.owner,
            due_date=t.due_date.isoformat() if t.due_date else None,
            priority=t.priority,
            status=t.status,
            source_quote=t.source_quote,
        ))
    wallet.credits -= 1  # only charged after a successful extraction
    db.commit()
    return {
        "chat_id": chat.id,
        "summary": result.summary,
        "topics": result.topics,
        "task_count": len(result.tasks),
        "credits": wallet.credits,
    }


@app.get("/api/chats")
def list_chats(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Chat, func.count(Task.id))
        .outerjoin(Task, Task.chat_id == Chat.id)
        .group_by(Chat.id)
        .order_by(Chat.created_at.desc())
    ).all()
    return [
        {
            "id": c.id,
            "filename": c.filename,
            "created_at": c.created_at.isoformat(),
            "task_count": n,
            "summary": c.summary,
            "topics": json.loads(c.topics) if c.topics else [],
        }
        for c, n in rows
    ]


@app.get("/api/chats/{chat_id}/tasks")
def chat_tasks(chat_id: int, db: Session = Depends(get_db)):
    chat = db.get(Chat, chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    return [task_to_dict(t) for t in sorted(chat.tasks, key=lambda t: t.id)]


@app.patch("/api/tasks/{task_id}")
def update_task(task_id: int, body: StatusUpdate, db: Session = Depends(get_db)):
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    task.status = body.status
    db.commit()
    return task_to_dict(task)


# ---------- Billing (Stripe Checkout) ----------

@app.get("/api/billing/balance")
def balance(client_id: str, db: Session = Depends(get_db)):
    return {"credits": get_wallet(db, client_id).credits}


@app.post("/api/billing/checkout")
def create_checkout(body: CheckoutRequest, request: Request, db: Session = Depends(get_db)):
    require_stripe()
    get_wallet(db, body.client_id)
    base = str(request.base_url).rstrip("/")
    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "unit_amount": PACK_PRICE_CENTS,
                    "product_data": {"name": f"{PACK_CREDITS} chat extractions"},
                },
                "quantity": 1,
            }],
            client_reference_id=body.client_id,
            metadata={"credits": str(PACK_CREDITS)},
            success_url=base + "/?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=base + "/",
        )
    except stripe.StripeError as e:
        raise HTTPException(502, f"Stripe error: {e.user_message or e}")
    return {"url": session.url}


@app.get("/api/billing/confirm")
def confirm_checkout(session_id: str, db: Session = Depends(get_db)):
    require_stripe()
    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except stripe.StripeError as e:
        raise HTTPException(400, f"Stripe error: {e.user_message or e}")
    if session.payment_status != "paid":
        raise HTTPException(402, "Payment not completed")
    wallet = get_wallet(db, session.client_reference_id)
    if not db.get(Payment, session.id):  # credit each session only once
        credits = int(session.metadata.get("credits", PACK_CREDITS))
        db.add(Payment(session_id=session.id, client_id=wallet.client_id,
                       credits=credits, amount_cents=session.amount_total or 0))
        wallet.credits += credits
        db.commit()
    return {"client_id": wallet.client_id, "credits": wallet.credits}
