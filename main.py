import os
from config import PDF_PATH
from doc_loader import extract_text_from_pdf
from chroma_store import store_pdf_in_chroma, is_pdf_already_stored
from rag_query import query_with_rag

def main():
    # Load and store PDF in Chroma if not already present
    if os.path.exists(PDF_PATH):
        if not is_pdf_already_stored("sample_pdf"):
            print(f"📄 Loading PDF: {PDF_PATH}")
            pdf_text = extract_text_from_pdf(PDF_PATH)
            store_pdf_in_chroma(pdf_text, "sample_pdf")
        else:
            print("✅ PDF already stored in ChromaDB")
    else:
        print("⚠ No PDF found at", PDF_PATH)
        return

    # Query loop
    while True:
        q = input("\n❓ Enter your question (or 'exit'): ")
        if q.lower() == "exit":
            break
        answer = query_with_rag(q)
        print(f"💡 Answer: {answer}")

if __name__ == "__main__":
    main()
