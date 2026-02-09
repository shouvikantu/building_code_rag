import os
import asyncio
import json
import threading
import logging
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status # Important import
from supabase_store import store_rag_result, maybe_clear_local_cache

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
_vercel_env = os.getenv("VERCEL") or os.getenv("VERCEL_ENV")
WORKING_DIR = os.getenv(
    "RAG_WORKING_DIR",
    "/tmp/lightrag_storage" if _vercel_env else str(BASE_DIR / "lightrag_storage"),
)
DOC_DIR = Path(os.getenv("RAG_DOC_DIR", str(BASE_DIR / "docs")))

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)

# Initialize LightRAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=openai_embed,
    chunk_token_size = 1100,
    chunk_overlap_token_size = 150,
    llm_model_max_async = 16,
    embedding_func_max_async = 16,
    max_parallel_insert = 10
)

_initialized = False
_init_lock = None

_loop = None
_loop_thread = None
_loop_ready = threading.Event()

logger = logging.getLogger(__name__)


def _run_loop() -> None:
    global _loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _loop = loop
    _loop_ready.set()
    loop.run_forever()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop_thread
    if _loop_thread is None or not _loop_thread.is_alive():
        _loop_ready.clear()
        _loop_thread = threading.Thread(
            target=_run_loop,
            name="lightrag-loop",
            daemon=True,
        )
        _loop_thread.start()
        _loop_ready.wait()
    return _loop


def _run_coroutine(coro):
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


async def ingest_documents():
    """Reads PDFs and inserts them into the graph."""
    from pypdf import PdfReader
    if not DOC_DIR.exists():
        logger.warning("Docs directory not found: %s", DOC_DIR)
        return
    for file_path in DOC_DIR.glob("*.pdf"):
        print(f"Extracting and Graphing: {file_path.name}...")
        reader = PdfReader(file_path)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if text.strip():
            await rag.ainsert(text)

async def main():
    # These two lines initialize the async locks that are currently failing
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    # Check if we have documents to ingest (first run only)
    # If the storage folder is empty, run ingestion
    if not any(Path(WORKING_DIR).iterdir()):
        await ingest_documents()
    
    # q = " Is day care allowed in this zone?  Provide a list of properties zoned to allow day care centers within area X of the City of Portland. Does the building have an E or I-4 occupancy permit? Does the property have open space, including parking lot, that could be converted into the required play area for 40 preschool children? How many classrooms will this building support assuming 25 students per classroom and 20 percent of the square footage offices, etc.?  Is the building on this property vacant? Does the building have the required number of ingress and egress places? Does the building have sprinklers? Does the building have a food preparation area? "
    q= "Is daycare allowed in an RM2 zone?"
    print(f"\nQuerying: {q}")
    
    # Perform a hybrid search (Graph + Vector)
    result = await rag.aquery(q, param=QueryParam(mode="hybrid"))
    
    print("\n=== GRAPH RAG ANSWER ===\n")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())


async def ensure_ready() -> None:
    """Initialize storages and ingest documents once per process."""
    global _initialized, _init_lock
    if _initialized:
        return
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    async with _init_lock:
        if _initialized:
            return
        await rag.initialize_storages()
        await initialize_pipeline_status()
        if not any(Path(WORKING_DIR).iterdir()):
            await ingest_documents()
        _initialized = True


def build_property_prompt(question: str, property_data: dict) -> str:
    """Combine property data with the user question for RAG retrieval."""
    property_blob = json.dumps(property_data, ensure_ascii=True, sort_keys=True)
    return (
        "You are answering a question about a specific property.\n"
        "Property information (authoritative):\n"
        f"{property_blob}\n\n"
        f"Question: {question}"
    )


async def aquery_property(question: str, property_data: dict) -> str:
    """Async wrapper for querying RAG with property context."""
    await ensure_ready()
    q = build_property_prompt(question, property_data)
    result = await rag.aquery(q, param=QueryParam(mode="hybrid"))
    await asyncio.to_thread(store_rag_result, question, property_data, result, "hybrid")
    await asyncio.to_thread(maybe_clear_local_cache, WORKING_DIR)
    return result


def query_property(question: str, property_data: dict) -> str:
    """Sync wrapper for Flask usage."""
    return _run_coroutine(aquery_property(question, property_data))


def start_background_init() -> None:
    """Kick off LightRAG initialization in the background."""
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(ensure_ready(), loop)

    def _log_error(fut):
        try:
            fut.result()
        except Exception:
            logger.exception("LightRAG initialization failed")

    future.add_done_callback(_log_error)
