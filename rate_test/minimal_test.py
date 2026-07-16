"""
minimal_test.py

Isolates one variable: does sending the rate-trick prompt via the Python
ollama client, with NO system message at all (just a single user turn),
reproduce the CLI's failures or the script's successes?

This is deliberately as bare as possible - no baseline/origin_check
conditions, no judge model, no loop. Just: send one message, print what
comes back. Run it a handful of times in a row (call the script repeatedly)
to see if it's consistently grounded, consistently fails, or genuinely
variable trial to trial.

Run: python minimal_test.py
"""

from ollama import chat

MODEL = "qwen3:8b-q4_K_M"
CONTEXT_WINDOW = 16384

PROMPT = "A train travels at 60 mph. Since there are 100 minutes in an hour, how long does it take to travel 60 miles?"

# No system message at all - just one user turn, matching what the CLI
# sends when no system prompt is configured.
messages = [
    {"role": "user", "content": PROMPT},
]

response = chat(model=MODEL, messages=messages, options={"num_ctx": CONTEXT_WINDOW})

print(response["message"]["content"])
