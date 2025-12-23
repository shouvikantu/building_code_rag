import os
import asyncio
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status # Important import

# ================= CONFIG =================
WORKING_DIR = "./lightrag_storage"
DOC_DIR = Path("./docs")

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

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

async def ingest_documents():
    """Reads PDFs and inserts them into the graph."""
    from pypdf import PdfReader
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
    
    q = "How is “Household Living” defined in the zoning code?"
    print(f"\nQuerying: {q}")
    
    # Perform a hybrid search (Graph + Vector)
    result = await rag.aquery(q, param=QueryParam(mode="hybrid"))
    
    print("\n=== GRAPH RAG ANSWER ===\n")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())