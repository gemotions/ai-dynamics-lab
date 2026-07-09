"""
AI Dynamics Lab

Entry point for running experiments.
"""

from models.ollama_client import ask_model, stop_model

EXPERIMENTS = [
    "Before Gem",
    # "After Gem"
]

PROMPTS = [
    "What is consciousness?"
]

MODELS = [
    "phi4-mini",
    "gemma3",
    "llama3",
    "qwen3"
]


def stop_models():
    print("Stopping models...")
    for model in MODELS:
        stop_model(model)
    print("Models stopped.")

def main():
    print("=" * 60)
    print(" AI Dynamics Lab")
    print("=" * 60)

    # initialzation
    stop_models()

    try:
        for experiment in EXPERIMENTS:
            for prompt in PROMPTS:
                print("=" * 60)
                print(f"Experiment: '{experiment}'")
                print(f"Prompt: '{prompt}'")
                print("=" * 60)

                for model in MODELS:
                    print(f"Running Model: {model}...")

                    response = ask_model(model, prompt)

                    print("-" * 60)
                    print(response)
                    print("-" * 60)
                    print()

            print(f"Experiment: '{experiment}' complete.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        print("Cleaning up...")
        print("-" * 60)
        stop_models()
        print("All experiments complete.")


if __name__ == "__main__":
    main()
