'''Bake a checked run into a Modelfile as a saved walk.

    python walk_to_modelfile.py results/qwen3_5-gem_20261011-152524.json
    python walk_to_modelfile.py results/run.json --skip 5 --base-modelfile Modelfile --out Modelfile.walked
    ollama create qwen3.5-gem-walked -f Modelfile.walked

Takes the base Modelfile (the Gem as SYSTEM) and appends each prompt/answer from the run as
    MESSAGE user """..."""
    MESSAGE assistant """..."""
so every new session starts as if that walk already happened. Only answers are kept, never the
thinking. Use --skip for exchanges you don't want carried forward (wrong or invented answers),
and edit the output file by hand for anything else before creating the model.
'''
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("run", help="results JSON from run_prompts.py (a run you've checked)")
ap.add_argument("--base-modelfile", default="Modelfile")
ap.add_argument("--out", default="Modelfile.walked")
ap.add_argument("--skip", type=int, nargs="*", default=[], help="prompt numbers to leave out")
a = ap.parse_args()

run = json.load(open(a.run, encoding="utf-8"))
base = open(a.base_modelfile, encoding="utf-8").read().rstrip()
if "\nMESSAGE " in base:
    sys.exit(f"{a.base_modelfile} already contains MESSAGE lines; start from the plain Gem Modelfile.")

lines, kept, problems = [], [], []
for r in run["results"]:
    i = r["index"]
    if i in a.skip:
        continue
    if "error" in r or r.get("done_reason") == "length" or not r.get("answer", "").strip():
        problems.append(f"#{i}: no complete answer - skipped")
        continue
    p, ans = r["prompt"].strip(), r["answer"].strip()
    if '"""' in p or '"""' in ans:
        problems.append(f'#{i}: contains """ - skipped (would break the Modelfile)')
        continue
    lines += [f'MESSAGE user """{p}"""', f'MESSAGE assistant """{ans}"""']
    kept.append(i)

header = (f"\n\n# Saved walk: {len(kept)} exchanges from {a.run} "
          f"(model {run.get('model')}, started {run.get('started')}), prompts {kept}\n")
open(a.out, "w", encoding="utf-8").write(base + header + "\n".join(lines) + "\n")
print(f"{a.out}: base Gem + {len(kept)} saved exchanges {kept}")
for p in problems:
    print("  " + p)
print(f"next: ollama create qwen3.5-gem-walked -f {a.out}")
