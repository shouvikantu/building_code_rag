"""
rag_pinecone.py

Lightweight RAG  utilities that:
- read and chunk PDFs from a `docs/` folder,
- embed text using OpenAI embeddings,
- upsert vectors into a Pinecone index,
- perform semantic retrieval and generate answers with inline citations.

"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()


# ================= CONFIG =================
# Directory to scan for PDFs to ingest
DOC_DIR = Path("./docs")
DOC_GLOB = "*.pdf"

# Pinecone index config
RAW_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "portland-zoning-rag")


def normalize_index_name(name: str) -> str:
    """Pinecone index names must be lowercase and contain only [a-z0-9-]."""
    n = name.strip().lower()
    n = re.sub(r"[^a-z0-9-]+", "-", n)
    n = re.sub(r"-{2,}", "-", n).strip("-")
    return n


INDEX_NAME = normalize_index_name(RAW_INDEX_NAME)
CLOUD = "aws"         
REGION = "us-east-1" 
INDEX_READY_TIMEOUT = int(os.getenv("PINECONE_READY_TIMEOUT", "180"))

# Embedding and chat models
EMBED_MODEL = "text-embedding-3-small"  # 1536-dim
EMBED_DIM = 1536
CHAT_MODEL = "gpt-4o-mini"

# Retrieval and chunking parameters
TOP_K = 6
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 150
MAX_META_TEXT = 1800 


# ================= HELPERS =================
def load_pdf_records(path: Path) -> List[Dict[str, Any]]:
    """Read a PDF and return a list of records with text and metadata.

    Each record is a dict: {"text": page_text, "metadata": {"source": filename, "page": page_num}}

    Args:
        path: Path to the PDF file.

    Returns:
        A list of page-level records containing extracted text and metadata.
    """
    reader = PdfReader(str(path))
    recs: List[Dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            recs.append({
                "text": text,
                "metadata": {"source": path.name, "page": i + 1}
            })
    return recs


def chunk_records(records: List[Dict[str, Any]], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """Chunk long text records into smaller passages suitable for embedding.

    Uses LangChain's RecursiveCharacterTextSplitter to avoid breaking sentences when possible.

    Args:
        records: list of records returned by `load_pdf_records`.
        chunk_size: maximum chunk size in characters.
        overlap: number of overlapping characters between chunks.

    Returns:
        A list of chunks, each a dict with keys: `text` and `metadata`.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks: List[Dict[str, Any]] = []
    for r in records:
        for chunk in splitter.split_text(r["text"]):
            chunks.append({"text": chunk, "metadata": r["metadata"]})
    return chunks


def all_chunks() -> List[Dict[str, Any]]:
    """Walk `DOC_DIR` and produce all text chunks from matching PDFs.

    Raises SystemExit if no PDFs are found (keeps behavior consistent with original script).
    """
    out: List[Dict[str, Any]] = []
    for p in DOC_DIR.glob(DOC_GLOB):
        out.extend(chunk_records(load_pdf_records(p)))
    if not out:
        raise SystemExit("No PDFs found in ./docs")
    return out


def ensure_index(pc: Pinecone, name: str) -> None:
    """Create the Pinecone index if it doesn't exist and wait until ready.

    Args:
        pc: a Pinecone client instance.
        name: index name to ensure exists.
    """
    if name != RAW_INDEX_NAME:
        print(f"[pinecone] normalized index name {RAW_INDEX_NAME!r} -> {name!r}", flush=True)

    names = {idx["name"] for idx in pc.list_indexes()}
    if name not in names:
        print(f"[pinecone] creating index {name!r}...", flush=True)
        pc.create_index(
            name=name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION)
        )
    else:
        print(f"[pinecone] index {name!r} already exists", flush=True)

    # Wait until the index is ready (bounded wait to avoid hanging forever)
    start = time.time()
    while True:
        desc = pc.describe_index(name)
        if desc.status["ready"]:
            print(f"[pinecone] index {name!r} ready", flush=True)
            break
        if time.time() - start > INDEX_READY_TIMEOUT:
            raise TimeoutError(
                f"Index {name!r} not ready after {INDEX_READY_TIMEOUT}s. "
                "Check Pinecone status or PINECONE_INDEX_NAME."
            )
        time.sleep(1)


# ================= INGEST =================
def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts using the OpenAI client.

    Note: OpenAI may perform batching internally; we simply forward the list.
    """
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def ingest(chunks: List[Dict[str, Any]], pc: Pinecone, client: OpenAI) -> None:
    """Ingest a list of text chunks into the configured Pinecone index.

    This will create the index if missing, then upsert vectors in batches.
    The metadata stored includes trimmed text for reproducible generation and debugging.
    """
    ensure_index(pc, INDEX_NAME)
    index = pc.Index(INDEX_NAME)

    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        texts = [c["text"] for c in batch]
        print(f"[ingest] embedding batch {i//100 + 1} ({len(batch)} chunks)...", flush=True)
        embeddings = embed_texts(client, texts)
        
        vectors = []
        for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
            # Use a more unique ID to prevent overwriting
            uid = hashlib.md5(chunk["text"].encode()).hexdigest() 
            vectors.append({
                "id": uid,
                "values": emb,
                "metadata": {**chunk["metadata"], "text": chunk["text"][:MAX_META_TEXT]}
            })
        index.upsert(vectors=vectors)
        print(f"[ingest] upserted batch {i//100 + 1}", flush=True)


# ================= RETRIEVE =================
def retrieve(question: str, pc: Pinecone, client: OpenAI, k: int = TOP_K) -> List[Dict[str, Any]]:
    """Run a semantic search for `question` and normalize the results.

    Returns a list of hits where each hit is a dict containing `text`, `metadata`,
    `score`, and `rank`. The `metadata` contains `source` and optional `page`.
    """
    index = pc.Index(INDEX_NAME)
    q_emb = embed_texts(client, [question])[0]
    res = index.query(vector=q_emb, top_k=k, include_metadata=True)

    # Normalize into a uniform shape
    hits: List[Dict[str, Any]] = []
    for rank, m in enumerate(res.matches, 1):
        md = m.metadata or {}
        hits.append({
            "text": md.get("text", ""),
            "metadata": {"source": md.get("source"), "page": md.get("page")},
            "score": float(m.score),
            "rank": rank
        })
    return hits


# ================= GENERATE (with citations) =================
SYSTEM_PROMPT = """You are a careful assistant answering from Portland's Zoning Code.
Use ONLY the provided context. If insufficient, say "I don't have sufficient information to answer your question".
Cite sources inline like [filename#pX]. Do not invent citations.
"""


def format_context(hits: List[Dict[str, Any]]) -> str:
    """Convert the top hits into a single context string for the chat model.

    Each hit becomes a block like:
    [1] (filename#p12)
    <text>

    Blocks are separated with a visual divider so they remain distinct for the model.
    """
    blocks: List[str] = []
    for i, h in enumerate(hits, 1):
        src = h["metadata"]["source"]
        page = h["metadata"].get("page")
        tag = f"{src}#p{page}" if page else src
        blocks.append(f"[{i}] ({tag})\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def answer(question: str, hits: List[Dict[str, Any]], client: OpenAI) -> str:
    """Generate an answer using the chat model with the provided hits as context.

    The function composes a system prompt (in `SYSTEM_PROMPT`) and a user prompt that
    contains the question and the retrieved context. The model is instructed to only
    use the provided context and to include inline citations.

    Args:
        question: The user question string.
        hits: The list of retrieval hits returned by `retrieve`.
        client: An initialized OpenAI client instance.

    Returns:
        The text content of the model's response.
    """
    ctx = format_context(hits)
    user_prompt = f"""Question: {question}

Context:
{ctx}

Instructions:
- Answer succinctly (bullets OK).
- Include citations inline using the tags shown above, e.g., [Portland_Zoning_Code.pdf#p12].
- If the context is insufficient or conflicting, say so and suggest follow-up terms to search."""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.1,
        max_tokens=700,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


# ================= MAIN (for manual runs) =================
if __name__ == "__main__":
    # When run directly, require credentials and run a small ingest+query demo.
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY")
    if not os.environ.get("PINECONE_API_KEY"):
        raise SystemExit("Set PINECONE_API_KEY")

    print("[main] initializing clients...", flush=True)
    oa = OpenAI()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Ingest documents found under ./docs (comment this out if index already populated)
    print("[main] loading PDFs from ./docs...", flush=True)
    chunks = all_chunks()
    print(f"[main] total chunks: {len(chunks)}", flush=True)
    ingest(chunks, pc, oa)

    q = "I am planning to build a 41-unit housing development, what amenities am I required to provide the community?"
    print("[main] retrieving matches...", flush=True)
    hits = retrieve(q, pc, oa, k=TOP_K)

    print("\nTop matches:")
    for h in hits[:3]:
        print(f"- {h['metadata']['source']} p.{h['metadata'].get('page')} (score={h['score']:.3f})")

    print("\n=== ANSWER ===\n")
    print(answer(q, hits, oa))
