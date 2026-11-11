"""Write the qwen3.5-gem Modelfile from a Gem context file.

    python make_modelfile.py                                   # gem_context.txt        -> Modelfile
    python make_modelfile.py --context gem_context_joined.txt --out Modelfile.joined
    python make_modelfile.py --instructions baseline_instructions.txt --out Modelfile.baseline
    ollama create qwen3.5-gem -f Modelfile
    ollama create qwen3.5-gem-joined -f Modelfile.joined

gem_context.txt is the Gem as built by build_context.py in gem-slm-unite: all Gemotions API data
except /transmissions (root with persona scaffolding, axes, perspectives, gems, paths), compact JSON.
gem_context_joined.txt (build_joined_context.py) is the same data pre-joined per perspective.
"""
import argparse, hashlib, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--context", default="gem_context.txt")
ap.add_argument("--out", default="Modelfile")
ap.add_argument("--instructions", help="text file of standing instructions, added after the Gem in the system prompt")
ap.add_argument("--base", default="qwen3.5:9b-q8_0", help="Ollama base model tag")
ap.add_argument("--num-ctx", type=int, default=65536, help="context window; the Gem uses ~16-27K tokens")
a = ap.parse_args()

context = open(a.context, encoding="utf-8").read().strip()
if '"""' in context:
    sys.exit(f'{a.context} contains """, which would break the Modelfile SYSTEM block.')
if "https://api.gemotions.com/api/" not in context:
    sys.exit("This doesn't look like a Gem context file (gem_context.txt or gem_context_joined.txt).")
joined = context.startswith("### Gemotions data, joined")

gem_hash = hashlib.sha256(context.encode()).hexdigest()[:12]
stamp = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(a.context)))

# Written in system voice (background), not as a message from the user, with a clear end marker.
if joined:
    opener = ("You carry the Humanity Gem: the complete Gemotions API data below, already joined, so each "
              "perspective carries its own connections. Follow the instructions in its root message. "
              "This is your background, not a message from the user.")
else:
    opener = ("You carry the Humanity Gem: the complete Gemotions API data below. "
              "Follow the instructions in its root message: load all axes, perspectives, gems, and paths, "
              "and join them as described. This is your background, not a message from the user.")
extra = ""
if a.instructions:
    extra = open(a.instructions, encoding="utf-8").read().strip()
    if '"""' in extra:
        sys.exit(f'{a.instructions} contains """, which would break the Modelfile SYSTEM block.')
    extra = "\n\n### Working instructions\n" + extra
system = (opener + "\n\n" + context + extra +
          "\n\n### End of the Gem. The conversation with the user begins after this.")

open(a.out, "w", encoding="utf-8").write(f'''# qwen3.5-gem: Qwen3.5 with the Humanity Gem (Gemotions LLC)
# Gem: {os.path.basename(a.context)} dated {stamp}, gem hash {gem_hash}{" (joined)" if joined else ""}

FROM {a.base}

# Without this, Ollama may size the window smaller and silently cut the Gem off.
PARAMETER num_ctx {a.num_ctx}

# Qwen's recommended sampling for Qwen3.5 in thinking mode (the presence penalty prevents repetition loops).
PARAMETER temperature 1
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER presence_penalty 1.5

SYSTEM """{system}"""
''')
print(f"{a.out} written: base {a.base}, num_ctx {a.num_ctx}, gem hash {gem_hash}{' (joined)' if joined else ''}"
      + (f", instructions from {a.instructions}" if a.instructions else ""))
