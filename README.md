# Portland Zoning Query Tool

A Flask web app that queries Portland zoning, building, and property information using the PortlandMaps API, with a RAG-powered Q&A feature backed by LightRAG and OpenAI.

## Architecture

- **`app.py`** — Flask web app with routes for address/ZIP lookup and RAG queries
- **`zoning.py`** — PortlandMaps API client (geocoding, zoning, building, taxlot queries)
- **`light_rag_impl.py`** — LightRAG wrapper for document ingestion and semantic Q&A
- **`templates/index.html`** — Web UI

## Setup

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set environment variables in `.env`:

```
OPENAI_API_KEY=...
```

Run the app:

```bash
python app.py
```

Open http://localhost:5001 in a browser.

### Docker (Google Cloud)

Build and run locally:

```bash
docker build -t building-codes .
docker run -p 5000:5000 --env-file .env building-codes
```

Deploy to Google Cloud Run:

```bash
gcloud run deploy building-codes --source . --region us-central1
```
