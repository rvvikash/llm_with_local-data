from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

models = client.models.list()

for m in models.data:
    if "embedding" in m.id.lower():
        print(m.id)
