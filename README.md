# DATA_storage_finder

## Chatbot improvements (RAG v2)

The chatbot now supports:

- Clarification-first flow for vague recommendation questions.
- Retrieval with relevance scores and confidence gating.
- Source-aware answering with citation IDs (`[S1]`, `[S2]`).
- Structured API response (`response`, `citations`, `confidence`, `route`, etc.).
- JSONL metrics logging in `logs/chat_metrics.jsonl`.
- Hybrid retrieval reranking (vector + keyword overlap + metadata boosts).
- Semantic factsheet chunking with section metadata (`section`, `source_trust`, `fetch_date`).

## Run evaluation

```bash
.venv/bin/python -m chatbot.evaluate_rag
```

Edit test prompts in `data/eval_questions.json`.

## Rebuild vector store (recommended after updates)

```bash
.venv/bin/python embed_textsaboutstorage_to_chroma.py
```

## Metrics dashboard

- UI: `/chat/metrics`
- API: `/api/chat/metrics?days=14`
- CLI summary:

```bash
.venv/bin/python -m chatbot.print_metrics
```

## Publish behind a domain name

The Flask app already listens on `0.0.0.0:5000`. To make it reachable via a domain like
`storagefinder.storagefinder.src.surf-hosted.nl`, configure the infrastructure layer to
forward that hostname to this app.

Recommended app env var:

```bash
export TRUSTED_HOSTS=storagefinder.storagefinder.src.surf-hosted.nl
```

What still needs to exist outside this repo:

- A DNS record for `storagefinder.storagefinder.src.surf-hosted.nl` pointing to the server IP.
- A reverse proxy / ingress / SURF-hosted route for that hostname forwarding to `http://127.0.0.1:5000`.
- Forwarded headers enabled (`Host`, `X-Forwarded-Proto`, `X-Forwarded-For`, `X-Forwarded-Port`).

## Production run

Do not use the Flask development server for public traffic.

1. Install/update dependencies:

```bash
.venv/bin/pip install -r requirements.txt
```

2. Create an environment file:

```bash
cp deploy/storagefinder.env.example deploy/storagefinder.env
```

3. Set at least:

```bash
SECRET_KEY=replace-this-with-a-long-random-secret
TRUSTED_HOSTS=storagefinder.storagefinder.src.surf-hosted.nl
APP_HOST=127.0.0.1
PORT=5000
FLASK_DEBUG=false
```

4. Run with Gunicorn:

```bash
.venv/bin/gunicorn --workers 2 --threads 4 --bind 127.0.0.1:5000 wsgi:application
```

5. Optional: install the provided systemd service from `deploy/storagefinder.service`.
