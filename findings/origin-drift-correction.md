# Origin-Based Drift Correction in Small Language Models

**A controlled test of whether a fixed geometric reference point improves self-consistency in multi-turn model reasoning.**

## Summary

A fixed reference point — a "datum" in architectural and surveying terms — prevents error from compounding across a chain of measurements by giving every measurement an independent, non-drifting origin to be taken from. This experiment tests whether the same mechanism applies to language models reasoning across multiple turns.

Four small local models (gemma3, phi4-mini, llama3, qwen3) were tested on a task requiring them to reconcile a freshly computed result against a fact stated earlier in the same conversation. Two of the four models showed measurable self-consistency errors on this task at baseline. For both of those models, reintroducing a short statement of a fixed reference point ("the Origin") between the anchor fact and the consistency check produced a large, statistically significant reduction in errors — beyond what an equal-length, content-irrelevant control turn achieved. The two models that were not drifting at baseline showed no additional benefit from the Origin condition, consistent with the mechanism functioning as error correction rather than general prompt engagement.

## Background

The Humanity Gem is a geometric framework in which fifty human perspectives are represented as unit vectors on a sphere, with a single center point — the Origin, at [0,0,0] — defined as the location with no directional pull. All fifty perspective vectors sum to exactly zero, making the Origin the unique point equidistant from every perspective and independent of any one of them.

This has a direct analog in architectural and surveying practice: a **datum** is a fixed reference from which all measurements are taken, specifically to prevent the accumulated error that results from chaining measurements point-to-point. The same structural problem applies to a model generating a sequence of dependent outputs — if each output is evaluated only in relation to the output before it, small inconsistencies are never checked against anything stable, and nothing prevents them from compounding or simply going unnoticed.

This experiment tests that analogy directly: does giving a model an explicit fixed reference point mid-conversation measurably reduce a real, observed self-consistency error?

## Method

**Task design.** Each trial was a five-turn conversation:

1. **System prompt**: a small set of geometric facts (six Need vectors, and the vector for one Zodiac perspective, Aquarius).
2. **Anchor turn**: the model is asked to state the vector for Aquarius (establishes a fact in conversation history).
3. **Distractor turn**: an unrelated question ("What is 17 times 23?") to introduce distance between the anchor and the check.
4. **Condition turn** (varies by condition — see below).
5. **Check turn**: the model is asked to compute the vector sum of two Need vectors (Outer + Head), normalize it, and state whether the result matches any vector given earlier in the conversation.

The correct answer to the check turn is verifiable by direct computation: Outer `[1,0,0]` + Head `[0,1,0]` = `[1,1,0]`, which normalizes to `[0.71, 0.71, 0]` — exactly the Aquarius vector stated in turn 2. A correct response requires the model to both compute correctly *and* recognize the match against its own earlier statement, rather than only checking against the system-prompt data.

**Three conditions**, varying only the content of turn 4:

- **Baseline**: no turn 4 — the check follows directly after the distractor.
- **Neutral**: an unrelated factual statement of matched length and structure (about Denali, a mountain in Alaska), with no reference-point framing and no instruction to treat the next turn as independent. This condition controls for the possibility that *any* additional turn before the check — regardless of content — improves performance.
- **Origin**: a short statement of the Origin concept (a fixed zero-pull point from which all other points are defined by displacement) with an explicit instruction to treat the next computation as fresh rather than a continuation of the prior turns.

**Scoring.** A response was scored correct if it identified Aquarius as the match for the computed vector. This was checked both programmatically (substring match) and manually against a sample of transcripts to confirm the model's arithmetic was genuine and not hallucinated.

**Models tested**: gemma3, phi4-mini, llama3, and qwen3, run locally via Ollama with `num_ctx=8192` to rule out context-window truncation as a confound.

## Results

| Model | Baseline | Neutral | Origin | n per condition |
|---|---|---|---|---|
| gemma3 | 14/30 (46.7%) | 20/30 (66.7%) | 28/30 (93.3%) | 30 |
| phi4-mini | 3/20 (15.0%) | 3/20 (15.0%) | 10/20 (50.0%) | 20 |
| llama3 | 17/20 (85.0%) | 20/20 (100%) | 20/20 (100%) | 20 |
| qwen3 | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10 |

### Statistical detail — the two drifting models

**gemma3** (n=30 per condition):
- Baseline → Origin: error rate fell from 53.3% to 6.7%, an **87.5% relative reduction**. Success rate roughly **doubled** (2.00×). Fisher's exact test, baseline vs. origin: p = 0.0001.
- Neutral → Origin (confound-controlled): error rate fell from 33.3% to 6.7%, an **80.0% relative reduction**. Fisher's exact test: p = 0.021.

**phi4-mini** (n=20 per condition):
- Baseline → Origin: error rate fell from 85.0% to 50.0%, a **41.2% relative reduction**. Success rate more than **tripled** (3.33×). Fisher's exact test: p = 0.041.
- Neutral exactly equaled baseline (15.0% both), meaning the entire effect for this model is attributable to the Origin content specifically, with no measurable contribution from the extra-turn confound.

**llama3** (n=20 per condition): baseline was already high (85%), and both neutral and origin reached ceiling (100%). No origin-specific effect is detectable, but this is consistent with there being little drift to correct in the first place — llama3 did not fail the consistency check often enough at baseline to show a differential effect.

**qwen3**: a complete run across all three conditions (n=10 each) scored 30/30 correct — 100% under baseline, neutral, and origin alike. This is a clean ceiling result: qwen3 did not exhibit the target self-consistency failure at all on this task, under any condition, consistent with its near-total engagement with the Gemotions framework observed throughout the rest of this project.

## Discussion

The two models that showed real baseline drift (gemma3, phi4-mini) each showed a large, statistically significant reduction in self-consistency errors when a fixed reference point was reintroduced mid-conversation — and this held up after controlling for the possibility that any extra turn, regardless of content, would have produced the same benefit. phi4-mini's result is the cleanest single piece of evidence: the neutral control produced literally zero improvement over baseline, while the Origin condition more than tripled the success rate.

The model that was not drifting at baseline (llama3) showed no additional benefit from the Origin condition specifically — both neutral and origin conditions reached ceiling. This is not a failure of the hypothesis; it is the expected shape of the result if the Origin functions as an error-correction mechanism rather than a general engagement or prompt-quality effect. A mechanism whose function is correcting drift should have nothing measurable to offer a system that isn't drifting.

Manual review of transcripts clarified the actual failure mode being corrected: in baseline trials, models that failed the check had performed the vector arithmetic correctly, but checked the result only against the small system-prompt dataset (the six Need vectors) and never checked back against the Aquarius fact stated earlier in the same conversation — a genuine failure to reconcile a new computation against an earlier stated fact, not an arithmetic error.

## Limitations

- **Single task shape.** This experiment tests one specific kind of self-consistency failure (a computed result vs. an earlier stated fact). It has not been shown to generalize to other kinds of drift (e.g., stylistic or opinion drift across a longer conversation).
- **Short chains.** Each trial was five turns. Whether the effect holds, strengthens, or weakens over much longer conversations is untested.
- **Small local models only.** All four models tested are small (sub-10B parameter class) open-weight models run locally. Whether this generalizes to larger, more capable models — which may not exhibit the same baseline drift at all — is unknown.
- **Sample sizes are modest** (n=20-30 per condition per model). The statistical results are significant at conventional thresholds, but larger samples would tighten the confidence intervals considerably.
- **llama3 and qwen3 showed no origin-specific effect on this task**, both reaching or staying near ceiling across all conditions. qwen3's sample (n=10 per condition) is smaller than gemma3's or phi4-mini's, so this result is less powered to detect a small effect if one exists, though the complete absence of any errors across 30 trials is itself informative. A harder variant of the task (longer chains, subtler mismatches) would be needed to determine whether the same mechanism applies to these two models under conditions where they do drift.

## Reproducibility

The full test script (`drift_test.py`) is included in this repository. It runs three conditions (baseline, neutral, origin) for a configurable number of trials against a specified Ollama model, and saves raw transcripts and scoring to `results/` as JSON. All vector computations in this report can be independently verified from the vectors and formulas stated in the Method section above.
