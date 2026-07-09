"""
models/ollama_client.py

Simple wrapper around the Ollama Python client.
"""

import subprocess
from ollama import chat


def stop_model(model: str):
    subprocess.run(
        ["ollama", "stop", model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def ask_model(model: str, prompt: str) -> str:
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

    # compute analytics of the response
    tokens_per_second = (
        response["eval_count"] /
        (response["eval_duration"] / 1_000_000_000)
    )

    # return response["message"]["content"]
    return {
        "model": model,
        "response": response["message"]["content"],
        "prompt_tokens": response.get("prompt_eval_count"),
        "response_tokens": response.get("eval_count"),
        "tokens_per_second": tokens_per_second,
        "total_duration": response.get("total_duration"),
        "load_duration": response.get("load_duration"),
        "eval_duration": response.get("eval_duration"),
    }
