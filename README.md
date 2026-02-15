# NYC Apartment Bot

A Telegram bot that monitors [StreetEasy](https://streeteasy.com) for NYC rental apartments matching your criteria, sends daily notifications with match scores, and helps you draft outreach messages to listing agents.

All interaction is natural language — tell the bot what you're looking for in plain English, and it figures out your preferences. No commands, no forms.

## How It Works

1. **Chat with the bot** on Telegram — describe what you're looking for ("2BR in the East Village, under $4k, need a dishwasher")
2. **The bot extracts your preferences** using Claude and confirms them with you
3. **Search on demand** or enable **daily scans** — the bot searches StreetEasy, scores each listing against your preferences, and sends the best matches
4. **Like, compare, and get details** on listings directly in Telegram
5. **Draft outreach messages** — the bot generates personalized messages for you to send to listing agents

### Listing Cards

Each listing shows up as a photo card with:
- Address, neighborhood, price, beds/baths, sqft
- Match score (0-100) with a visual bar
- Key pros and cons
- Inline buttons: Like, Pass, Details (Telegraph Instant View), StreetEasy link

### Scoring

Listings are scored by Claude against your full preference profile, including a `constraint_context` that tracks what's a dealbreaker vs. a nice-to-have. Say "I absolutely need a dishwasher" and it's a hard filter; say "gym would be nice" and it's a soft score boost.

## Architecture

```
Telegram ──webhook──▸ Modal (serverless)
                          │
                     Claude API (conversation + scoring)
                          │
                     Apify (StreetEasy scraper)
                          │
                     Telegraph (detail pages)
```

- **Runtime**: [Modal](https://modal.com) — serverless Python, handles webhooks + daily cron
- **Scraping**: [Apify](https://apify.com) actor (`memo23/apify-streeteasy-cheerio`) — Cheerio-based, residential proxies
- **Conversation**: Claude API with `tool_use` — every message goes through the LLM
- **Messaging**: Telegram Bot API
- **Detail pages**: Telegraph (Instant View in Telegram)
- **Models**: Pydantic v2
- **Language**: Python 3.11, fully async

### Why This Architecture

StreetEasy doesn't have a public API. The Apify actor uses StreetEasy's internal GraphQL endpoint with residential proxies. A two-layer filtering pipeline compensates for StreetEasy's RECOMMENDED sort ignoring neighborhood filters (~98% of results are out-of-area):

1. **Neighborhood pre-filter** — drops out-of-area results using an alias map
2. **LLM scoring** — Claude evaluates each surviving listing against all preferences in a single call

## Setup

### Prerequisites

- [Modal](https://modal.com) account
- [Apify](https://apify.com) account with the `memo23/apify-streeteasy-cheerio` actor
- [Anthropic](https://console.anthropic.com) API key
- Telegram bot (create via [@BotFather](https://t.me/BotFather))

### Configure Secrets

Set up three Modal secrets:

```bash
# Anthropic API key
modal secret create anthropic ANTHROPIC_API_KEY=sk-ant-...

# Telegram bot token
modal secret create streeteasy-telegram SE_TELEGRAM_BOT_TOKEN=...

# Apify API token
modal secret create apify APIFY_API_TOKEN=apify_api_...
```

### Deploy

```bash
pip install -r requirements.txt

# Deploy to Modal
modal deploy modal_app.py

# Set up Telegram webhook
modal run modal_app.py::setup
```

### Run Tests

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

## Project Structure

```
modal_app.py              # Modal entry points: webhook, daily cron, outreach sender
src/
  conversation.py         # LLM conversation engine — Claude API with 16 tools
  claude_client.py        # Async wrapper around Anthropic SDK
  scanner.py              # Search pipeline: fetch, dedupe, pre-filter, LLM score
  apify_scraper.py        # Apify actor wrapper for StreetEasy
  telegram_handler.py     # Telegram Bot API: webhooks, messages, callbacks
  formatter.py            # Telegram HTML formatting, listing cards, keyboards
  telegraph_pages.py      # Telegraph Instant View pages for listing details
  outreach.py             # Agent outreach draft generation and revision
  models.py               # Pydantic models: ChatState, Preferences, Listing, etc.
  storage.py              # JSON persistence on Modal Volume
  config.py               # NYC neighborhoods, amenities, constants
tests/
  test_conversation.py    # Conversation engine + all 16 tools (54 tests)
  test_scanner.py         # Scan pipeline + LLM scoring (31 tests)
  test_models.py          # Pydantic model validation (22 tests)
  test_formatter.py       # HTML formatting + listing cards (18 tests)
  test_storage.py         # Persistence + atomicity (6 tests)
  test_telegram_handler.py
```

## Conversation Tools

The bot has 16 tools available to Claude during conversation:

| Tool | What it does |
|------|-------------|
| `update_preferences` | Extract and save apartment preferences from natural language |
| `search_apartments` | Trigger a StreetEasy search |
| `show_preferences` | Display current preferences |
| `mark_ready` | Enable daily automated scans |
| `pause_daily_scans` | Pause daily scans |
| `get_liked_listings` | Show all liked listings |
| `get_listing_details` | Full details for a specific listing |
| `compare_listings` | Side-by-side comparison of 2-5 listings |
| `draft_outreach` | Generate a message to send to a listing agent |
| `update_current_apartment` | Store info about your current apartment (used for outreach context) |
| `clear_search_history` | Reset seen listings so searches show everything again |
| `reset_preferences` | Start over with fresh preferences |
| ...and more | |

## Cost

For personal use (1-2 users, 1 daily scan):

| Service | ~Monthly Cost |
|---------|--------------|
| Modal | ~$1-2 (within free tier) |
| Anthropic API | ~$5-15 |
| Apify actor | $19 (rental) |
| Telegram | Free |

## License

MIT
