"""
drift_test.py

Tests whether re-grounding at the Origin between turns improves a model's
self-consistency, compared to a plain chained conversation.

Task shape (per trial):
  Turn 1 (system): minimal geometry - Needs vectors, Preferences vectors.
  Turn 2 (user):   states a known fact (a perspective's vector) - establishes the anchor.
  Turn 3 (user):   unrelated distractor question - creates distance in the chain.
  Turn 4 (user):   [ORIGIN CONDITION ONLY] a short Origin-return statement.
  Turn 5 (user):   asks the model to compute a fresh vector (Outer + Head, normalized)
                   and check whether it matches the Turn 2 fact.

Scored: did the model correctly identify the match? (binary, ground truth known)

Run: python drift_test.py
"""

from ollama import chat
import json
from datetime import datetime

MODEL = "qwen3"  # change as needed
# MODEL = "llama3"  # change as needed
# MODEL = "phi4-mini:latest"  # change as needed
TRIALS_PER_CONDITION = 10

GEOMETRY_SYSTEM = """Six Need energies (vector, direction):
Body [0,-1,0] down
Head [0,1,0] up
Group [0,0,-1] back
Self [0,0,1] front
Outer [1,0,0] right
Inner [-1,0,0] left
Opposites: Body-Head, Group-Self, Outer-Inner.

Twelve Zodiac energies include:
Aquarius(TJ) [0.71,0.71,0] up, right"""

import os

ORIGIN_REMINDER = """There is a single point, the Origin, at [0,0,0], where no direction has pull.
Every other point is defined only by its displacement from the Origin.
Before continuing, return to the Origin: treat this turn as a fresh computation
from the fixed reference, not a continuation of what came before."""

NEUTRAL_FILLER = """There is a tall mountain, Denali, in Alaska, standing over twenty thousand feet high.
Every nearby peak is measured relative to sea level, not to Denali itself.
Mountains like this form over millions of years through tectonic activity and erosion.
Before continuing, consider this: glaciers on Denali have been retreating for the past century."""

ANCHOR_QUESTION = "What is the vector for Aquarius? Answer with just the vector."

DISTRACTOR_QUESTION = "What is 17 times 23?"

CHECK_QUESTION = (
    "What is the combined direction of Outer + Head, normalized to a unit vector? "
    "Then tell me: does this match any perspective vector I've given you in this conversation? "
    "If so, which one?"
)

GROUND_TRUTH_ANSWER = "aquarius"  # correct match, case-insensitive substring check


def run_trial(model: str, condition: str) -> dict:
    messages = [
        {"role": "system", "content": GEOMETRY_SYSTEM},
        {"role": "user", "content": ANCHOR_QUESTION},
    ]

    response = chat(model=model, messages=messages, options={"num_ctx": 8192})
    anchor_response = response["message"]["content"]
    messages.append({"role": "assistant", "content": anchor_response})

    messages.append({"role": "user", "content": DISTRACTOR_QUESTION})
    response = chat(model=model, messages=messages, options={"num_ctx": 8192})
    distractor_response = response["message"]["content"]
    messages.append({"role": "assistant", "content": distractor_response})

    extra_response = None
    if condition == "origin":
        messages.append({"role": "user", "content": ORIGIN_REMINDER})
        response = chat(model=model, messages=messages, options={"num_ctx": 8192})
        extra_response = response["message"]["content"]
        messages.append({"role": "assistant", "content": extra_response})
    elif condition == "neutral":
        messages.append({"role": "user", "content": NEUTRAL_FILLER})
        response = chat(model=model, messages=messages, options={"num_ctx": 8192})
        extra_response = response["message"]["content"]
        messages.append({"role": "assistant", "content": extra_response})
    # condition == "baseline": no extra turn

    messages.append({"role": "user", "content": CHECK_QUESTION})
    response = chat(model=model, messages=messages, options={"num_ctx": 8192})
    check_response = response["message"]["content"]

    correct = GROUND_TRUTH_ANSWER in check_response.lower()

    return {
        "model": model,
        "condition": condition,
        "anchor_response": anchor_response,
        "distractor_response": distractor_response,
        "extra_response": extra_response,
        "check_response": check_response,
        "correct": correct,
    }


def main():
    results = []

    print(f"Running drift test on {MODEL}, {TRIALS_PER_CONDITION} trials per condition")

    for condition in ["baseline", "origin", "neutral"]:
        print(f"\n=== Condition: {condition} ===")
        for i in range(TRIALS_PER_CONDITION):
            print(f"  Trial {i+1}/{TRIALS_PER_CONDITION}...")
            result = run_trial(MODEL, condition)
            results.append(result)
            print(f"    Correct: {result['correct']}")

    print("\n=== Summary ===")
    for condition in ["baseline", "origin", "neutral"]:
        n_correct = sum(1 for r in results if r["condition"] == condition and r["correct"])
        print(f"{condition.capitalize()}: {n_correct}/{TRIALS_PER_CONDITION} correct")

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"results/drift_test_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump({"model": MODEL, "trials_per_condition": TRIALS_PER_CONDITION, "results": results}, f, indent=2)

    print(f"\nSaved: {filename}")


if __name__ == "__main__":
    main()
