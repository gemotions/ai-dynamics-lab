"""
AI Dynamics Lab

Entry point for running experiments.
"""

from models.ollama_client import ask

MODELS = [
    "phi4-mini",
    "gemma3",
    "llama3",
    "qwen3"
]

PROMPT = "What is consciousness?"


def main():
    print("=" * 60)
    print(" AI Dynamics Lab")
    print("=" * 60)

    for model in MODELS:
        print(f"Running {model}...")

        response = ask(model, PROMPT)

        print("-" * 60)
        print(response)
        print("-" * 60)
        print()

if __name__ == "__main__":
    main()
