from light_rag_impl import query_rag
import logging

# Configure logging to capture any errors
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Testing RAG query...")
    try:
        # A simple query. The index might be empty if build_index.py didn't finish, 
        # but it shouldn't crash with "bound to different event loop".
        response = query_rag("What are the zoning requirements?")
        print(f"RAG Response: {response}")
    except Exception as e:
        print(f"RAG Query Failed: {e}")
        exit(1)
