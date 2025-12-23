# filename: rag_portland_code_min.py
# pip install chromadb pypdf langchain-text-splitters openai
# export OPENAI_API_KEY=your_key

import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
try:
    from chromadb.config import Settings
except Exception:
    # Fallback Settings for editors/static analyzers that can't resolve chromadb.config
    from dataclasses import dataclass
    @dataclass
    class Settings:
        anonymized_telemetry: bool = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from openai import OpenAI

# ----------------- CONFIG -----------------
DOC_DIR = Path("docs")
DOC_GLOB = "*.pdf"           # place the zoning code PDF here
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"
TOP_K = 6

# ----------------- LOAD & CHUNK -----------------
def load_pdf_records(path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(path))
    recs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            recs.append({"text": text, "metadata": {"source": path.name, "page": i+1}})
    return recs

def chunk_records(records, chunk_size=1100, overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for r in records:
        for chunk in splitter.split_text(r["text"]):
            chunks.append({"text": chunk, "metadata": r["metadata"]})
    return chunks

# ----------------- VECTOR STORE (Chroma, in-memory) -----------------
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.create_collection(name="portland_code", metadata={"hnsw:space": "cosine"})

def embed_texts(texts: List[str], client_oa: OpenAI):
    # OpenAI will batch under the hood
    resp = client_oa.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def ingest_docs(client_oa: OpenAI):
    all_chunks = []
    for p in DOC_DIR.glob(DOC_GLOB):
        all_chunks += chunk_records(load_pdf_records(p))

    texts = [c["text"] for c in all_chunks]
    metas = [c["metadata"] for c in all_chunks]
    emb   = embed_texts(texts, client_oa)
    ids   = [f"chunk-{i}" for i in range(len(texts))]
    collection.add(documents=texts, metadatas=metas, embeddings=emb, ids=ids)

# ----------------- RETRIEVE -----------------
def retrieve(question: str, client_oa: OpenAI, k: int = TOP_K):
    q_emb = embed_texts([question], client_oa)[0]
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents","metadatas","distances"])
    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": text, "metadata": meta, "score": 1 - dist})  # cosine sim approx
    return hits

# ----------------- GENERATE (with inline citations) -----------------
SYSTEM_PROMPT = """You are a careful assistant answering from Portland's Zoning Code.
Use ONLY the provided context. If insufficient, say you don't know.
Cite sources inline like [filename#pX]. Do not invent citations."""

def format_context(hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        src = h["metadata"]["source"]
        page = h["metadata"].get("page")
        tag = f"{src}#p{page}" if page else src
        blocks.append(f"[{i}] ({tag})\n{h['text']}")
    return "\n\n---\n\n".join(blocks)

def ask_llm(question: str, hits, client_oa: OpenAI):
    context = format_context(hits)
    user_prompt = f"""Question: {question}

Context:
{context}

Instructions:
- Answer succinctly in bullet points when helpful.
- Include citations inline using the (tag) shown above, e.g., [Portland_Zoning_Code.pdf#p12].
- If the answer isn't clearly supported by the context, say you don't know and mention which terms to search next."""
    resp = client_oa.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.1,
        max_tokens=600,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":user_prompt},
        ]
    )
    return resp.choices[0].message.content

# ----------------- MAIN -----------------
if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY")

    oa = OpenAI()
    ingest_docs(oa)

    question = "I am planning to build a 41-unit housing development, what amenities am I required to provide the community?"
    hits = retrieve(question, oa, k=TOP_K)
    print("\nTop matches (for debugging):")
    for h in hits[:3]:
        print(f"- {h['metadata']['source']} p.{h['metadata'].get('page')} (score={h['score']:.3f})")

    print("\n=== ANSWER ===\n")
    print(ask_llm(question, hits, oa))
