Local RAG web UI

What I added
- `app.py`: small Flask app that provides a text box to ask a question and shows the returned answer.
- `templates/index.html`: very small UI.
- `requirements.txt`: packages needed.

How to run

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Ensure environment variables are set (you must have credentials):

```bash
export OPENAI_API_KEY="..."
export PINECONE_API_KEY="..."
```

3. Run the app:

```bash
python app.py
```

Open http://127.0.0.1:5000 in a browser.

Notes and assumptions
- This app uses the retrieval and answer functions defined in `rag_pinecone.py` (it expects those functions and names to exist).
- The original script also creates/ingests the Pinecone index if necessary; this app assumes the index already exists and is populated. If not, run `python rag_pinecone.py` once to ingest documents.
- This is intentionally minimal and for local use only.
