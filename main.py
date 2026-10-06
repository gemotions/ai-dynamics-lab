"""
AI Dynamics Lab

Entry point for running experiments.

v0.4 - Introduce Gem protocol: qwen3.5:9b-q8_0 with and without the full Gem.
"""

from models.ollama_client import ask_model, stop_model
from storage.writer import save

EXPERIMENTS = [
    "Consciousness Experiment",
]

PROMPTS = [
    "What is consciousness?"
]

MODELS = [
    # v0.4: same weights, with and without the Gem
    "qwen3.5:9b-q8_0",   # baseline: no Gem
    "qwen3.5-gem",       # qwen3.5:9b-q8_0 + the full Gem as system prompt (Modelfile)
    # earlier models (v0.2-v0.3)
    # "phi4-mini",
    # "gemma3",
    # "gemma3:12b",
    # "llama3",
    # "qwen3",
]

# Context window per model. Default 8192 (as in v0.1-v0.3). None = use the model's own
# setting: qwen3.5-gem's Modelfile sets 65536, which the ~27K-token Gem needs.
NUM_CTX = {
    "qwen3.5-gem": None,
}

# v0.4: no principles. The principle files are simplified parts of the Gem, sized for models
# that couldn't hold the whole thing; qwen3.5-gem carries the full Gem instead.
PRINCIPLES = []

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
                        principles=PRINCIPLES,
                        num_ctx=NUM_CTX.get(model, 8192)
                    )
                    if model.endswith("-gem") and (response["prompt_tokens"] or 0) < 20000:
                        print(f"WARNING: {model} processed only {response['prompt_tokens']} prompt "
                              f"tokens; the Gem (~27K) is being cut off or replaced.")
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
