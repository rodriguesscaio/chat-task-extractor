# Chat Task Extractor

Upload a chat export (.txt, e.g. WhatsApp) and get the actionable tasks extracted by Claude, stored in SQLite and shown on a dashboard.

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_MODEL=claude-sonnet-5-5
export STRIPE_SECRET_KEY=sk_test_...   # Stripe Dashboard > Developers > API keys (test mode)
uvicorn main:app --reload
```

Open http://127.0.0.1:8000 and upload `sample_chat.txt` to test.

## Stripe: pay-per-use credits

Each browser gets 3 free extractions (`FREE_CREDITS` env var). After that, **Buy 10 more ($5)** opens Stripe Checkout.
Pay with test card `4242 4242 4242 4242`, any future expiry date, any CVC. Back on the dashboard, the app checks the
Checkout Session with Stripe and adds the credits. Each session can only be credited once.
An extraction only uses a credit if it succeeds.

No products need to be set up in the Stripe Dashboard. The price is created inline (`price_data`).

## API

| Method | Path | Description |
|---|---|---|
| POST | `/api/chats` | Upload a .txt file (form field `file`), run extraction, returns `chat_id` + summary |
| GET | `/api/chats` | List chats (id, filename, created_at, task count) |
| GET | `/api/chats/{id}/tasks` | Tasks for one chat |
| PATCH | `/api/tasks/{id}` | Body `{"status": "open" \| "done"}` |
| GET | `/api/billing/balance?client_id=` | Credits left |
| POST | `/api/billing/checkout` | Body `{"client_id"}`, returns the Stripe Checkout URL |
| GET | `/api/billing/confirm?session_id=` | Verifies payment with Stripe and adds credits |

## Notes

- Only the last 30,000 characters of a chat are sent to the LLM.
- If the LLM returns invalid JSON (or JSON that fails Pydantic validation), it retries once, then returns HTTP 502.
- Data is stored in `tasks.db` (created automatically).
