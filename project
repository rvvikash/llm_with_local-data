# doc\_rag\_project — PDF RAG (ChromaDB + Local Embeddings + OpenRouter LLM)

This repository contains a ready-to-paste project that:

* Extracts text and images from a PDF
* Computes local embeddings with `sentence-transformers` (free, offline)
* Stores embeddings persistently in ChromaDB
* Performs retrieval and answers questions using OpenRouter (GPT-4o) as the LLM
* Shows related images (extracted from PDF pages) when relevant

> **Security**: embeddings remain local. Only retrieved text chunks are sent to OpenRouter.

---

## File structure (copy these files into a GitHub repo as-is)

```
doc_rag_project/
├── data/                      # place your PDF here (example: sample.pdf)
├── chroma_store/              # persistent chroma files (ignored by git)
├── pdf_images/                # images extracted from PDF
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── config.py
├── pdf_loader.py
├── chroma_store.py
├── rag_query.py
├── main.py
```

---

> **Important:** Put your PDF in `data/` (e.g. `data/pyspark.pdf`) and set your OpenRouter API key in a `.env` file (see `.env.example`).

---

## `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
.env

# ChromaDB persistent store (local vectors) — remove if you want to include it
chroma_store/

# extracted images
pdf_images/

# macOS
.DS_Store
```

---

## `.env.example`

```
# Copy to .env and update
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## `requirements.txt`

```
chromadb
sentence-transformers
pymupdf
openai
python-dotenv
```

---

## `config.py`

```python
# ===================== CONFIGURATION =====================
# Copy .env.example -> .env and set OPENROUTER_API_KEY there
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Path to input PDF (relative to repo root)
PDF_PATH = "data/sample.pdf"

# Chroma collection name
COLLECTION_NAME = "my_pdf_docs"

# Persistent storage folder for Chroma
CHROMA_DB_PATH = "chroma_store"

# Folder for extracted PDF images
PDF_IMAGES_DIR = "pdf_images"

# Chunk size (in words) used to split pages into chunks
CHUNK_SIZE = 400
```

---

## `pdf_loader.py`  (extracts text + images using PyMuPDF)

```python
import os
import fitz  # PyMuPDF
from config import PDF_IMAGES_DIR


def load_pdf_with_images(pdf_path):
    """Return list of (page_num, text) and a dict page_num -> [image_paths].

    - page_num is 0-indexed
    - images are saved to PDF_IMAGES_DIR
    """
    os.makedirs(PDF_IMAGES_DIR, exist_ok=True)
    pdf_document = fitz.open(pdf_path)

    pages_text = []
    page_images_map = {}

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text = page.get_text("text") or ""
        pages_text.append((page_num, text))

        image_list = page.get_images(full=True)
        if not image_list:
            continue

        page_image_paths = []
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            pix = fitz.Pixmap(pdf_document, xref)
            img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num+1}_img_{img_index}.png"
            img_path = os.path.join(PDF_IMAGES_DIR, img_filename)

            if pix.n < 5:  # RGB or grayscale
                pix.save(img_path)
            else:  # CMYK: convert
                pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(img_path)

            page_image_paths.append(img_path)
            pix = None

        if page_image_paths:
            page_images_map[page_num] = page_image_paths

    pdf_document.close()
    return pages_text, page_images_map
```

---

## `chroma_store.py`  (local sentence-transformers embeddings + persistent ChromaDB)

```python
import os
import chromadb
from sentence_transformers import SentenceTransformer
from config import COLLECTION_NAME, CHROMA_DB_PATH, CHUNK_SIZE

# Ensure the DB folder exists
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Create or get collection
if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
    collection = chroma_client.get_collection(COLLECTION_NAME)
else:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

# Local embedding model (small & fast)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def store_pdf_in_chroma(pages_text, doc_name):
    """pages_text: list of (page_num, text)
    doc_name: a short identifier (e.g. 'pyspark')
    """
    for page_num, page_text in pages_text:
        if not page_text or not page_text.strip():
            continue
        chunks = chunk_text(page_text)
        for idx, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()

            # id format includes page and chunk so we can trace back
            doc_id = f"{doc_name}_p{page_num}_c{idx}"
            collection.add(
                ids=[doc_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"page": page_num, "doc": doc_name}]
            )


def is_pdf_already_stored(doc_name):
    """Check whether the first chunk id exists in collection."""
    some_id = f"{doc_name}_p0_c0"
    try:
        res = collection.get(ids=[some_id])
        return len(res.get("ids", [])) > 0
    except Exception:
        return False


def get_collection_and_model():
    return collection, embedding_model
```

---

## `rag_query.py`  (retrieves from Chroma and calls OpenRouter LLM)

```python
from openai import OpenAI
from chroma_store import get_collection_and_model
from config import OPENROUTER_API_KEY

# Initialize OpenRouter client for chat/LLM answers
llm_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

collection, embedding_model = get_collection_and_model()


def query_with_rag(question, k=3):
    # 1) embed query locally
    q_emb = embedding_model.encode(question).tolist()

    # 2) retrieve top-k
    results = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
    docs = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0]

    print("\n📄 Retrieved PDF Chunks (top k):")
    for i, (d, m) in enumerate(zip(docs, metadatas), start=1):
        print(f"--- chunk {i} (page={m.get('page')}) ---")
        print(d[:400].replace('\n', ' '))
        print()

    context = "\n\n".join(docs)

    # 3) call LLM (OpenRouter) with context
    response = llm_client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer using only the provided context. If not present, say 'Not found in document.'"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=700
    )

    # Access content safely
    answer = response.choices[0].message.content

    # collect pages that had images later by checking metadatas
    pages = sorted({m.get("page") for m in metadatas if m.get("page") is not None})

    return answer, pages
```

---

## `main.py` (CLI to ingest PDF and ask questions)

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # place early to avoid warnings

from config import PDF_PATH, PDF_IMAGES_DIR
from pdf_loader import load_pdf_with_images
from chroma_store import store_pdf_in_chroma, is_pdf_already_stored
from rag_query import query_with_rag


def main():
    # If PDF_PATH not set or missing, prompt user
    if not os.path.exists(PDF_PATH):
        pdf_path = input("Enter path to PDF file: ").strip()
    else:
        pdf_path = PDF_PATH

    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Extract text + images
    print(f"📄 Loading PDF: {pdf_path}")
    pages_text, page_images_map = load_pdf_with_images(pdf_path)

    # Save page_images_map to disk (optional) by leaving images in PDF_IMAGES_DIR
    print("Images extracted to:", PDF_IMAGES_DIR)

    # Store embeddings if not present
    if not is_pdf_already_stored(doc_name):
        print("💾 Storing PDF in ChromaDB...")
        store_pdf_in_chroma(pages_text, doc_name)
    else:
        print("✅ PDF already stored in ChromaDB")

    # interactive query loop
    while True:
        q = input("\n❓ Enter your question (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break

        ans, pages = query_with_rag(q, k=3)
        print("\n📝 Answer:\n", ans)

        # if any pages have images, print image paths
        if pages:
            image_folder = PDF_IMAGES_DIR
            print("\n🖼 Related pages with (possible) images:")
            for p in pages:
                # list images that match page prefix
                prefix = f"{doc_name}_page_{p+1}_"
                matches = [f for f in os.listdir(image_folder) if f.startswith(prefix)] if os.path.exists(image_folder) else []
                if matches:
                    for m in matches:
                        print(" -", os.path.join(image_folder, m))
                else:
                    print(f" - page {p+1}: (no extracted images)" )


if __name__ == "__main__":
    main()
```

---

## How to use

1. Clone or create repo and paste these files.
2. `cp .env.example .env` and set `OPENROUTER_API_KEY`.
3. Put your PDF under `data/` (or provide path when prompted).
4. Create a virtualenv and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

5. Run:

```bash
python main.py
```

6. Ask questions. The script prints retrieved chunks and shows related image file paths (if any).

---

## Confirming the PDF is used

* The `rag_query.py` prints the top-k retrieved chunks (first 400 chars) from the PDF before calling the LLM — if you see your PDF text there, answers are coming from your PDF.

---

## Notes & Next steps

* If you want *all* processing fully local (no OpenRouter usage), you can replace `llm_client` with a local LLM (e.g. `text-generation-webui`, `llama.cpp` wrappers, or Ollama) and call it instead.
* If your PDF is large, increase `CHUNK_SIZE` or tune `k` in `query_with_rag`.

---

If you want, I can also export the architecture diagram (the visual you asked for) into `docs/architecture.png` ready to commit. Let me know and I will add it.
