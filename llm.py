import os

from openai import OpenAI


def ask_qwen(question: str, context: str) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        return (
            "Demo mode: no DASHSCOPE_API_KEY is configured.\n\n"
            "The retrieval step worked successfully. "
            "Configure the API key to let Qwen generate the final answer.\n\n"
            f"Retrieved context preview:\n{context[:800]}"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )

    model = os.getenv("QWEN_MODEL", "qwen-plus")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a paper question-answering assistant. "
                    "Answer only from the supplied context. "
                    "If the context is insufficient, say that the document "
                    "does not provide enough information."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or ""
