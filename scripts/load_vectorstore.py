"""
Enhanced ChromaDB Loader with HuggingFace Embeddings
----------------------------------------------------
- Ensures correct model loading
- Adds error handling
- Prints vector store stats
- Ready to plug into retriever/QA pipelines
"""

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# === CONFIG ===
PERSIST_DIR = "vectorstore/chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # explicit full model path

def load_vectorstore(persist_dir: str = PERSIST_DIR, model_name: str = EMBED_MODEL):
    """Reconnect to existing Chroma vector DB with HuggingFace embeddings."""
    
    # Check if vector store exists
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"Vector store directory '{persist_dir}' not found!")

    print(f"Using embeddings model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Reconnect to Chroma
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # Debug info
    try:
        count = vectorstore._collection.count()
        print(f"Vector store loaded successfully with {count} documents.")
    except Exception as e:
        print("Warning: Could not fetch collection stats:", e)

    return vectorstore

if __name__ == "__main__":
    # Load vector store
    vs = load_vectorstore()

    # Quick test query
    query = "salinity profiles near Indian Ocean"
    results = vs.similarity_search(query, k=2)

    print("\n Sample query:", query)
    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}:")
        print("Content:", doc.page_content[:250], "...")
        print("Metadata:", doc.metadata)
