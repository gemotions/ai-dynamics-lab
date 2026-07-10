"""
AI Dynamics Lab

Entry point for running experiments.
"""

from models.ollama_client import ask_model, stop_model
from storage.writer import save

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

run_results = []


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
                    response = ask_model(
                        experiment,
                        model,
                        prompt, 
                        principles=[
                            "01-line.md",
                            "02-line.md",
                            "03-line.md",
                            "04-origin.md"
                        ]
                    )
                    run_results.append(response)
                    print(f"Model {model} complete.")

            print(f"Experiment: '{experiment}' complete.")


    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        print("Cleaning up...")
        print("-" * 60)
        stop_models()
        print("All experiments complete.")


    # store the results of this run
    save(results=run_results)


if __name__ == "__main__":
    main()
