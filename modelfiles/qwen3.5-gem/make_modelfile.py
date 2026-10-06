"""Write the qwen3.5-gem Modelfile from gem_context.txt.

    python make_modelfile.py
    ollama create qwen3.5-gem -f Modelfile

gem_context.txt is the Gem as built by build_context.py in gem-slm-unite: all Gemotions API data
except /transmissions (root with persona scaffolding, axes, perspectives, gems, paths), compact JSON.
To update the Gem, rebuild gem_context.txt there, copy it here, and run this again.
"""
import argparse, hashlib, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--context", default="gem_context.txt")
ap.add_argument("--base", default="qwen3.5:9b-q8_0", help="Ollama base model tag")
ap.add_argument("--num-ctx", type=int, default=65536, help="context window; the Gem uses ~27K tokens")
a = ap.parse_args()

context = open(a.context, encoding="utf-8").read().strip()
if '"""' in context:
    sys.exit('gem_context.txt contains """, which would break the Modelfile SYSTEM block.')
if "### https://api.gemotions.com/api/" not in context:
    sys.exit("This doesn't look like a gem_context.txt from build_context.py.")

gem_hash = hashlib.sha256(context.encode()).hexdigest()[:12]
stamp = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(a.context)))
system = ("Here is the complete Gemotions API data. Follow the instructions in the root message: "
          "load all axes, perspectives, gems, and paths, and join them as described.\n\n" + context)

open("Modelfile", "w", encoding="utf-8").write(f'''# qwen3.5-gem: Qwen3.5 with the Humanity Gem (Gemotions LLC)
# Gem: gem_context.txt dated {stamp}, gem hash {gem_hash}

FROM {a.base}

# The Gem uses ~27K tokens; without this, Ollama may size the window smaller and silently cut the Gem off.
PARAMETER num_ctx {a.num_ctx}

# Qwen's recommended sampling for Qwen3.5 in thinking mode (the presence penalty prevents repetition loops).
PARAMETER temperature 1
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER presence_penalty 1.5

SYSTEM """{system}"""
''')
print(f"Modelfile written: base {a.base}, num_ctx {a.num_ctx}, gem hash {gem_hash}")
