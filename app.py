"""
app.py

Very small Flask web UI that exposes a single page where users can ask a
question and receive a RAG-generated answer. The view is deliberately simple
and intended for local use only.

Expected environment variables:
- OPENAI_API_KEY
- PINECONE_API_KEY

The app imports `retrieve` and `answer` from `rag_pinecone.py` and uses them
to obtain the top hits and produce a final answer.
"""

from flask import Flask, render_template, request
import os
import traceback
import markdown
# Import the retrieval/answer functions from the RAG module
from rag_pinecone import retrieve, answer, TOP_K

from openai import OpenAI
from pinecone import Pinecone


app = Flask(__name__)


def make_clients():
    """Create and return initialized OpenAI and Pinecone clients.

    Raises:
        RuntimeError: if required environment variables are missing.

    Returns:
        Tuple(OpenAI, Pinecone)
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    if not os.environ.get("PINECONE_API_KEY"):
        raise RuntimeError("PINECONE_API_KEY not set")
    oa = OpenAI()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"]) 
    return oa, pc


@app.route("/", methods=["GET", "POST"])
def index():
    answer_text = None
    error = None
    question = ""
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            error = "Please enter a question."
        else:
            try:
                oa, pc = make_clients()
                hits = retrieve(question, pc, oa, k=6)
                raw_answer = answer(question, hits, oa)
                
                # Convert Markdown to HTML
                # 'fenced_code' and 'tables' are good extensions for RAG
                answer_text = markdown.markdown(raw_answer, extensions=['fenced_code', 'tables'])
                
            except Exception as e:
                error = f"Error: {e}"
    return render_template("index.html", answer=answer_text, error=error, question=question)

if __name__ == "__main__":
    # Default to localhost:5000. Debug mode is helpful for local development.
    app.run(host="127.0.0.1", port=5000, debug=True)
