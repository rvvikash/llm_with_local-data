from openai import OpenAI
from chroma_store import get_collection
from config import OPENROUTER_API_KEY

# Get Chroma collection and local embedding model
collection, embedding_model = get_collection()

# OpenRouter client for answering
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

def query_with_rag(question):
    # 1. Embed question locally
    q_embedding = embedding_model.encode(question).tolist()

    # 2. Search Chroma for relevant docs
    results = collection.query(query_embeddings=[q_embedding], n_results=3)

    context = "\n".join(results["documents"][0])

    # 3. Ask OpenRouter LLM
    response = llm_client.chat.completions.create(
        model="openai/gpt-4o-mini",  # fast & cheap
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the context to answer."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content
