"""
models/ollama_client.py

Simple wrapper around the Ollama Python client.
"""

from pathlib import Path
from ollama import chat
import subprocess


def stop_model(model: str):
    subprocess.run(
        ["ollama", "stop", model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def ask_model(experiment: str, model: str, prompt: str, principles=None, num_ctx=8192) -> dict:
    """
    Send a prompt to an Ollama model and return its response.

    Args:
        experiment: The experiment name (e.g. "Before Gem")
        model: The model name (e.g. "gemma3")
        prompt: The user's prompt
        principles: Principle files sent as a system message. Leave empty for
            models that carry their own system prompt (e.g. qwen3.5-gem):
            a system message here would replace the Gem.
        num_ctx: Context window to request. 8192 matches v0.1-v0.3. Pass None
            to use the model's own setting (qwen3.5-gem's Modelfile sets 65536;
            8192 would cut the ~27K-token Gem off).

    Returns:
        The model's response as a string.
    """

    messages = []

    if principles:
        system_text = ""

        for principle in principles:
            system_text += load_principle(principle)
            system_text += "\n\n"

        messages.append({
            "role": "system",
            "content": system_text
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    print(f"Messages: {messages}")

    options = {} if num_ctx is None else {"num_ctx": num_ctx}

    response = chat(
        model=model,
        messages=messages,
        options=options
    )

    # compute analytics of the response
    tokens_per_second = (
        response["eval_count"] /
        (response["eval_duration"] / 1_000_000_000)
    )

    # return response["message"]["content"]
    return {
        "experiment": experiment,
        "prompt": prompt,
        "model": model,
        "unique_key": f"{experiment}:{prompt}:{model}",
        "response": response["message"]["content"],
        "thinking": getattr(response["message"], "thinking", None),
        "num_ctx": num_ctx,
        "prompt_tokens": response.get("prompt_eval_count"),
        "response_tokens": response.get("eval_count"),
        "tokens_per_second": tokens_per_second,
        "total_duration": response.get("total_duration"),
        "load_duration": response.get("load_duration"),
        "eval_duration": response.get("eval_duration"),
    }

def load_principle(filename):
    return Path(f"principles/{filename}").read_text()
