"""Run a list of prompts from a text file through an Ollama model and save everything.

    python run_prompts.py prompts.txt
    python run_prompts.py prompts.txt --model qwen3.5-gem --conversation
    python run_prompts.py prompts.txt --seed 1 --max-tokens 8000 --out results/run1.json

Prompt file format:
    - Prompts are separated by a blank line, so a prompt can span several lines.
    - Lines starting with # are comments and are skipped.

Modes:
    default         each prompt in a FRESH session (no memory between prompts; clean comparisons)
    --conversation  all prompts in ONE session, in order (each prompt sees the earlier turns)
    --interactive   when the prompts finish, keep talking to the model yourself; your turns are
                    saved to the same results file (marked "interactive")

Saved for every prompt: the prompt, thinking, answer, churn (characters of thinking),
prompt tokens (the Gem is ~26K, so this confirms it loaded), output tokens, and timings.
Results are written after every prompt, so an interruption (Ctrl-C) loses nothing finished.

Settings come from the Modelfile (num_ctx, sampling). This script deliberately does not set num_ctx,
which would override the Modelfile and could cut the Gem off.
"""
import argparse, json, os, sys, time
from ollama import chat

ap = argparse.ArgumentParser()
ap.add_argument("prompts", help="text file of prompts, separated by blank lines")
ap.add_argument("--model", default="qwen3.5-gem")
ap.add_argument("--conversation", action="store_true", help="one continuous session instead of fresh sessions")
ap.add_argument("--seed", type=int, help="fixed seed for repeatable runs")
ap.add_argument("--max-tokens", type=int, help="cap on output tokens per reply (stops runaway thinking)")
ap.add_argument("--no-think", action="store_true", help="turn thinking off")
ap.add_argument("--quiet", action="store_true", help="don't stream text to the screen")
ap.add_argument("--interactive", action="store_true",
                help="after the prompts, keep talking to the model (continues the conversation; /quit to stop)")
ap.add_argument("--out", help="results file (default: results/<model>_<timestamp>.json)")
a = ap.parse_args()


def read_prompts(path):
    blocks, cur = [], []
    for line in open(path, encoding="utf-8"):
        if line.lstrip().startswith("#"):
            continue
        if line.strip() == "":
            if cur:
                blocks.append("\n".join(cur).strip())
                cur = []
        else:
            cur.append(line.rstrip("\n"))
    if cur:
        blocks.append("\n".join(cur).strip())
    return [b for b in blocks if b]


prompts = read_prompts(a.prompts)
if not prompts:
    sys.exit(f"No prompts found in {a.prompts}")

options = {}
if a.seed is not None:
    options["seed"] = a.seed
if a.max_tokens:
    options["num_predict"] = a.max_tokens

out = a.out or os.path.join("results", f"{a.model.replace('/', '_').replace(':', '-')}_{time.strftime('%Y%m%d-%H%M%S')}.json")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
run = {"model": a.model, "prompts_file": a.prompts, "mode": "conversation" if a.conversation else "fresh",
       "seed": a.seed, "max_tokens": a.max_tokens, "think": not a.no_think,
       "started": time.strftime("%Y-%m-%d %H:%M:%S"), "results": []}
history = []          # the running conversation (conversation mode, and interactive)
last_exchange = []    # the most recent prompt + answer (lets --interactive follow a fresh-mode run)


def ask(i, prompt, base, label):
    """Send one prompt on top of `base` messages. Returns (result, messages_with_answer) or (None, None)."""
    messages = base + [{"role": "user", "content": prompt}]
    print(f"\n{'=' * 70}\n[{label}] {prompt}\n{'=' * 70}", flush=True)
    t0 = time.time()
    thinking, answer, final = "", "", None
    try:
        for chunk in chat(model=a.model, messages=messages, stream=True, think=not a.no_think,
                          options=options or None, keep_alive="30m"):
            th, tx = chunk.message.thinking or "", chunk.message.content or ""
            if th:
                if not thinking and not a.quiet:
                    print("--- thinking ---", flush=True)
                thinking += th
            if tx:
                if not answer and not a.quiet:
                    print("\n--- answer ---", flush=True)
                answer += tx
            if not a.quiet:
                sys.stdout.write(th + tx); sys.stdout.flush()
            if chunk.done:
                final = chunk
    except Exception as e:
        print(f"\n[error on prompt {i}: {e}]")
        run["results"].append({"index": i, "prompt": prompt, "error": str(e)})
        json.dump(run, open(out, "w"), indent=1, ensure_ascii=False)
        return None, None

    secs = round(time.time() - t0, 1)
    ns = lambda x: round((x or 0) / 1e9, 1)
    result = {
        "index": i, "prompt": prompt, "thinking": thinking, "answer": answer,
        "churn": len(thinking), "answer_chars": len(answer),
        "prompt_tokens": getattr(final, "prompt_eval_count", None),
        "output_tokens": getattr(final, "eval_count", None),
        "done_reason": getattr(final, "done_reason", None),
        "seconds": secs, "load_seconds": ns(getattr(final, "load_duration", 0)),
    }
    run["results"].append(result)
    json.dump(run, open(out, "w"), indent=1, ensure_ascii=False)

    note = " (hit token cap)" if result["done_reason"] == "length" else ""
    print(f"\n[{secs}s | prompt tokens {result['prompt_tokens']} | churn {result['churn']:,} chars | "
          f"output {result['output_tokens']} tokens{note}]")
    # A reply cut off by --max-tokens would sit in the history half-finished and confuse later
    # turns ("was my answer cut off? is this my turn or the prompt?"). Record it as a clear marker.
    kept = answer if result["done_reason"] != "length" else \
        "[No complete answer: the reply was cut off at the token limit.]"
    if result["done_reason"] == "length":
        print("[cut off at the token limit - recorded in the conversation as an unanswered turn]")
    return result, messages + [{"role": "assistant", "content": kept}]


print(f"{a.model}: {len(prompts)} prompts, {run['mode']} sessions -> {out}")
try:
    for i, prompt in enumerate(prompts, 1):
        result, convo = ask(i, prompt, history if a.conversation else [], f"{i}/{len(prompts)}")
        if convo:
            last_exchange = convo
            if a.conversation:
                history = convo
except KeyboardInterrupt:
    print("\n[interrupted - saving what's finished]")

print(f"\nDone: {len(run['results'])} of {len(prompts)} prompts saved to {out}")
print(f"{'#':>3}  {'churn':>8}  {'seconds':>7}  prompt")
for r in run["results"]:
    if "error" in r:
        print(f"{r['index']:>3}  {'error':>8}  {'':>7}  {r['prompt'][:60]}")
    else:
        print(f"{r['index']:>3}  {r['churn']:>8,}  {r['seconds']:>7}  {r['prompt'][:60]}")

if a.interactive:
    try:
        import readline  # noqa: F401  (arrow keys edit the line)
    except ImportError:
        pass
    convo = history if a.conversation else last_exchange
    where = "the whole conversation" if a.conversation else "the last prompt and answer"
    print(f"\n[interactive: continuing from {where}. Type a message; /quit to stop.]")
    i = len(run["results"])
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/q", "/quit", "/exit"):
            break
        i += 1
        try:
            result, new = ask(i, q, convo, f"you #{i}")
        except KeyboardInterrupt:
            print("\n[reply interrupted]")
            continue
        if result:
            result["interactive"] = True
            json.dump(run, open(out, "w"), indent=1, ensure_ascii=False)
            convo = new
    print(f"[interactive turns saved to {out}]")
