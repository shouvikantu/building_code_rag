import os
import logging
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
WORKING_DIR = os.getenv("RAG_WORKING_DIR", str(BASE_DIR / "lightrag_storage"))
DOC_DIR = Path(os.getenv("RAG_DOC_DIR", str(BASE_DIR / "docs")))

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)

if not os.path.exists(DOC_DIR):
    os.makedirs(DOC_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

# Global RAG instance
_rag_instance = None


async def get_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=gpt_4o_mini_complete,
            embedding_func=openai_embed,
            chunk_token_size=1200,
            chunk_overlap_token_size=50,
            llm_model_max_async=1,
            embedding_func_max_async=1,
        )
    return _rag_instance


import asyncio
import threading
import concurrent.futures

# Global background loop handling
_loop = None
_loop_thread = None
_loop_lock = threading.Lock()

def _loop_thread_target():
    global _loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _loop = loop
    loop.run_forever()

def get_bg_loop():
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is None:
            _loop_thread = threading.Thread(target=_loop_thread_target, daemon=True)
            _loop_thread.start()
            # Busy wait for loop to start (very fast)
            while _loop is None:
                pass
    return _loop

def _run_on_bg_loop(coro):
    """Submit coroutine to the background loop and wait for result."""
    loop = get_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()

async def _ingest_async():
    rag = await get_rag()
    # Initialize storages (REQUIRED)
    await rag.initialize_storages()
    
    from pypdf import PdfReader
    files = list(DOC_DIR.glob("*.pdf"))
    if not files:
        logger.warning(f"No PDFs found in {DOC_DIR}")
        return

    for file_path in files:
        logger.info(f"Processing {file_path.name}...")
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            
            if text.strip():
                # use ainsert for async insert
                await rag.ainsert(text)
                logger.info(f"Inserted {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to ingest {file_path.name}: {e}")

def ingest_docs():
    """Ingest all PDFs from the docs directory."""
    try:
        _run_on_bg_loop(_ingest_async())
    except Exception as e:
        logger.error(f"Async ingest failed: {e}")

async def _query_async(question: str, mode: str):
    rag = await get_rag()
    await rag.initialize_storages()
    # Reduced top_k to save memory
    param = QueryParam(
        mode=mode,
        top_k=20  # Reduced from default (often 60 or higher)
    )
    return await rag.aquery(question, param=param)

def query_rag(question: str, mode: str = "hybrid") -> str:
    """Query the RAG system."""
    return _run_on_bg_loop(_query_async(question, mode))

