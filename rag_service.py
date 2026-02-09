import os
import logging
from flask import Flask, request, jsonify
from light_rag_impl import query_property, start_background_init


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAG_API_TOKEN = os.getenv("RAG_API_TOKEN", "").strip()

start_background_init()


def _authorized(req) -> bool:
    if not RAG_API_TOKEN:
        return True
    auth = req.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.removeprefix("Bearer ").strip() == RAG_API_TOKEN
    return req.headers.get("X-API-Key", "").strip() == RAG_API_TOKEN


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/rag/query")
def rag_query():
    if not _authorized(request):
        return jsonify({"error": "unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    property_data = payload.get("property")

    if not question or not isinstance(property_data, dict):
        return jsonify({"error": "Invalid payload. Expected question and property."}), 400

    try:
        answer = query_property(question, property_data)
    except Exception as exc:
        logger.exception("RAG query failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
