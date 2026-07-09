"""
models/ollama_client.py

Simple wrapper around the Ollama Python client.
"""

from ollama import chat


def ask(model: str, prompt: str) -> str:
    """
    Send a prompt to an Ollama model and return its response.

    Args:
        model: The model name (e.g. "gemma3")
        prompt: The user's prompt

    Returns:
        The model's response as a string.
    """

    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # return response["message"]["content"]
    return {
        "model": model,
        "response": response["message"]["content"],
        "prompt_tokens": response.get("prompt_eval_count"),
        "response_tokens": response.get("eval_count"),
        "total_duration": response.get("total_duration"),
        "load_duration": response.get("load_duration"),
        "eval_duration": response.get("eval_duration"),
    }
