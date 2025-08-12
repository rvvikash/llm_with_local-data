import os
import chromadb
from sentence_transformers import SentenceTransformer
from config import COLLECTION_NAME, CHROMA_DB_PATH

os.makedirs(CHROMA_DB_PATH, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
    collection = chroma_client.get_collection(COLLECTION_NAME)
else:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

# Local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def store_pdf_in_chroma(pdf_text, doc_name):
    chunks = chunk_text(pdf_text)
    for idx, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(
            ids=[f"{doc_name}_{idx}"],
            documents=[chunk],
            embeddings=[embedding]
        )

def is_pdf_already_stored(doc_name):
    results = collection.get(ids=[f"{doc_name}_0"], include=["documents"])
    return len(results["ids"]) > 0

def get_collection():
    return collection, embedding_model
