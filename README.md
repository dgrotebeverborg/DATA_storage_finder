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

## Server installation

These steps match the working installation for `https://storagefinder.uu.nl/`.

Prerequisites:

- DNS for `storagefinder.uu.nl` must point to the server IP.
- `nginx`, `certbot`, and Python 3 must be installed on the server.
- Run the commands below from the project directory.

1. Create the virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

2. Create the environment file:

```bash
cp deploy/storagefinder.env.example deploy/storagefinder.env
```

3. Set at least these values in `deploy/storagefinder.env`:

```bash
SECRET_KEY=replace-this-with-a-long-random-secret
TRUSTED_HOSTS=storagefinder.storagefinder.src.surf-hosted.nl,storagefinder.uu.nl
APP_HOST=127.0.0.1
PORT=5000
FLASK_DEBUG=false
SESSION_COOKIE_SECURE=false
SESSION_COOKIE_SAMESITE=Lax
```

4. Verify Gunicorn can start:

```bash
.venv/bin/gunicorn --workers 2 --threads 4 --bind 127.0.0.1:5000 wsgi:application
```

5. Install the systemd service:

```bash
sudo cp deploy/storagefinder.service /etc/systemd/system/storagefinder.service
sudo systemctl daemon-reload
sudo systemctl enable --now storagefinder
sudo systemctl status storagefinder --no-pager
```

6. Install the nginx site:

```bash
sudo cp deploy/storagefinder.nginx.conf /etc/nginx/sites-available/storagefinder
sudo ln -s /etc/nginx/sites-available/storagefinder /etc/nginx/sites-enabled/storagefinder
```

If the symlink already exists, skip the `ln -s` command.

7. Request the TLS certificate:

```bash
sudo certbot certonly --nginx -d storagefinder.uu.nl --non-interactive --register-unsafely-without-email --agree-tos
```

8. Test and reload nginx:

```bash
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl status nginx --no-pager
```

## Notes

- The nginx site proxies directly to Flask on `127.0.0.1:5000`.
- The current nginx setup does not use the SRAM/SURF authentication include.
- After changing `deploy/storagefinder.env`, restart the app with `sudo systemctl restart storagefinder`.
- After changing `deploy/storagefinder.nginx.conf`, copy it to `/etc/nginx/sites-available/storagefinder`, run `sudo nginx -t`, and reload nginx.
