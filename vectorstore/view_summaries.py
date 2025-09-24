# view_summaries.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "vectorstore/chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Initialize embeddings + Chroma DB
# ----------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# ----------------------------
# View stored summaries
# ----------------------------
print(f"Total summaries stored: {db._collection.count()}")

docs = db._collection.get(include=["metadatas", "documents"])

for i, (doc, meta) in enumerate(zip(docs["documents"], docs["metadatas"])):
    print(f"\n--- Summary {i+1} ---")
    print("Text:", doc)
    print("Metadata:", meta)

    if i >= 9:  # Show only first 10
        break
