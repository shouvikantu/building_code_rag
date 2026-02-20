import os
import logging
from light_rag_impl import ingest_docs

# Configure logging to show progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print("Starting index build process...")
    try:
        ingest_docs()
        print("Index build completed successfully!")
        print(f"Index stored in: {os.path.abspath('lightrag_storage')}")
    except Exception as e:
        print(f"Failed to build index: {e}")
        exit(1)
