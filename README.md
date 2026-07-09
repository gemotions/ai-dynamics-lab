# AI Dynamics Lab

A research environment for exploring the behavior of local language models.

The lab is being built incrementally:

- Build the plumbing
- Establish baseline behavior
- Introduce protocols (e.g. Gem)
- Measure changes
- Iterate

---

## Setup

### Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run

```bash
(venv)% python main.py
```

---

## Current Models

- phi4-mini
- gemma3
- llama3
- qwen3

---

## Project Structure

```
ai-dynamics-lab/
│
├── main.py
├── config.py
│
├── models/
├── experiments/
├── protocols/
├── metrics/
├── prompts/
├── results/
└── tests/
```

---

## Experiment Log

| Version | Status | Notes |
|---------|--------|-------|
| v0.1 | ✅ | Connected to Ollama |
| v0.2 | ✅ | Multiple models running |
| v0.3 | ⬜ | Save baseline results |
| v0.4 | ⬜ | Introduce Gem protocol |

