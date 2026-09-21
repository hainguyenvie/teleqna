# TeleQnA closed-book 8B — status report

18/09/2026. Full detail: [`METHOD_STUDYKIT.md`](METHOD_STUDYKIT.md) · [`PLAN_CLOSED_BOOK_8B.md`](PLAN_CLOSED_BOOK_8B.md)

## Task

Submit a single Qwen3-8B that answers the GSMA Open Telco `teleqna` column (10,000 MCQ)
**closed-book**: no retrieval at inference, no second model above 8B, no training on the
benchmark rows. Test questions may be used only to select public source documents; the
answer key and `explanation` field are measurement instruments and never enter data
generation, filtering, or training.

Baseline 71.85. With eight retrieved windows in context the same model scores 83.75.
The whole track is the question of how much of that +11.9 can be moved into the weights.

## Result

| | |
|---|---|
| Current checkpoint | `models/kit/vd6/ep1` — **79.69** |
| Rotated-choice control (rot1) | 78.05 |
| recover / broke / unparsed | 1,220 / 435 / 0 |
| Harness-faithful (Inspect contract, vLLM server) | 79.07 on `vd3`, matching internal 79.04 |
| Serving | greedy, one forward pass, original harness parser |

Per subject, base → vd6: Standards specifications 61.15 → **71.45 (+10.30)**,
Standards overview 69.80 → 78.00, Research publications 74.73 → 81.84,
Research overview 74.15 → 80.80, Lexicon 83.60 → 92.20.

Standards specifications is the subject that did not move for prompt optimisation
(−2.80) or for thinking mode (+0.35). It is a knowledge gap, and it is where the
weights channel opened widest.

Reference points: telecom professionals 64.86 · GPT-4 (2023) 74.91 · gpt-5 83.80 (n≈1,000) ·
AT&T OTel-LLM-8.3B-QnA 91.20 (SOTA, n≈10,000).

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     1. EXTERNAL KNOWLEDGE                           │
│                                                                     │
│  TeleQnA questions                                                  │
│        │                                                            │
│        ▼                                                            │
│  BM25 + reranking          (questions only, never the answer key)   │
│        │                                                            │
│        ▼                                                            │
│  Telecom corpus                                                     │
│  3GPP + IEEE + TCC + Wikipedia          5.1M chunks                 │
│        │                                                            │
│        ▼                                                            │
│  Relevant source windows (~1.2k tokens)      114,125 windows        │
│                                                                     │
│  Functional coverage 89.4%   ·   ~520 questions have no source      │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   2. KNOWLEDGE AMPLIFICATION                        │
│                                                                     │
│                Each source window                                   │
│                        │                                            │
│        ┌───────────────┼─────────────────────────┐                  │
│        ▼               ▼                         ▼                  │
│    Verbatim        Fact views                QA / MCQ               │
│                    K paraphrases                                    │
│        │               │                         │                  │
│        └───────────────┴────────────┬────────────┘                  │
│                                     ▼                               │
│                               STUDY-KIT       1.41B tokens          │
│                                                                     │
│  + replay + anchors + masked reconstruction + chat bridge           │
│                                                                     │
│  Every gate is a string match, never an LLM judge                   │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   3. KNOWLEDGE INTERNALIZATION                      │
│                                                                     │
│                   Full-weight CPT                                   │
│                   8 x H200 · lr 3e-5 · 1 epoch · 2.20B tokens       │
│                                                                     │
│         Base Qwen3-8B                    Big CPT model              │
│            71.85%        ───────────────►     77.81%                │
│                                                                     │
│          Knowledge enters weights, but gain saturates               │
│          2.5x more data buys fewer broke, not more recover          │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     4. FAILURE DIAGNOSIS                            │
│                                                                     │
│     recover / broke · rot1 · letter probabilities · recall probe    │
│     oracle-union · calibration · source-in-context diagnosis        │
│                                                                     │
│                         Remaining errors                            │
│                               │                                     │
│             ┌─────────────────┼──────────────────┐                  │
│             ▼                 ▼                  ▼                  │
│       Missing source        NEVER             FLICKER               │
│         ~520 q            1,144 q             2,396 q               │
│                          confident            unstable              │
│                            wrong             prediction             │
│                               │                  │                  │
│                        Binding error     Consistency error          │
└─────────────────────────────────────────────────────────────────────┘
               │                   │                    │
               ▼                   ▼                    ▼
       Corpus / retrieval   Sibling-Contrast     Vote Distillation
                                   │                    │
                                   │               Soup / EMA
                                   │                    │
                                   └──────────┬─────────┘
                                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  5. TARGETED BEHAVIOR REFINEMENT                    │
│                                                                     │
│  Recall-QA              Sibling-Contrast         Consistency        │
│  "retrieve fact"       "bind correct sibling"  "stabilize choice"   │
│                                                                     │
│  entity → property      entity ↔ property        same content       │
│                                                across permutations  │
│                                                                     │
│              vd3 79.04 → sibling 79.66 → vd6 79.69                  │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          6. SERVING                                 │
│                                                                     │
│              Single Qwen3-8B · Closed-book · Greedy                 │
│              No retrieval · No voting · No teacher                  │
│                                                                     │
│                          79.69%                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## What the diagnosis found

The oracle union of 18 checkpoints is **88.56%**, close to the 89.4% source ceiling.
The pipeline has therefore already answered 88.6% of the test set correctly at some
point; 64.6% is correct in every checkpoint. The gap from 79.7 to 88.6 is a problem of
holding answers simultaneously, not of learning them.

The two error populations need different treatments:

- **flicker** (2,396) — margin median 0.50, gold ranked second on 82% of them.
  Treated by consistency training (vote distillation) and same-lineage weight averaging.
- **never** (1,144) — margin 0.90, gold ranked second on only 45%. The model is
  confidently wrong. 413 of 428 inspected cases had the gold string verbatim in a window
  and repeated at least five times in the kit. Reading samples showed one pattern: the
  model picks a **sibling entity from the same passage** (EDGE-1 → EDGE-3, state N8 → N9,
  first-order → second-order Markov, 10 cm → 1.5 m). This is an attribute-to-entity
  binding failure, not missing knowledge.

Sibling-contrast was built from that diagnosis: MCQs whose distractors are other siblings
**from the same window**, with every option and the evidence required verbatim in the
window. It is the largest single-stage gain since vote distillation (+0.62 / +0.79 rot1).

Extraction, not storage, is the bottleneck. On 1,214 questions the checkpoint had not
fixed, base + retrieved windows answers 83% correctly — the fact is recoverable from the
source, so the failure is in retrieving it from the weights.

## What did not work

| Attempt | Result |
|---|---|
| Filtering training windows by measured effect (gold-fix, answer-key upper bound) | 72.20 vs 72.64 label-free — worse |
| lr 5e-5 | recover 1,030, broke 1,041 — net zero |
| Letter-DPO at lr 3e-6 | **collapse to 14.11** — likelihood displacement on a one-token completion |
| Numeric drill (84,689 Q/A) | target bucket unchanged, 63.7 → 63.7 |
| Behaviour view (All-of-the-above / NOT) | skews the test prior (1:1 data vs 4.5:1 real) |
| MLP-only updates | −35 broke and −30 recover, net zero |
| Tier-3 re-study, 808M tokens | stopped at 26% of the run by pre-registered rule |
| Weight soup across different lineages | lands between parents, does not capture the union |
| Prompt-only recite-then-answer | −6.5 |

Three pack bugs were worth about 4 points combined: self-replay anchors that taught the
base's wrong beliefs (+1.50 when fixed), a chat/raw mode mismatch (+2.7 recoverable), and
a missing `<|im_end|>` that let the model write past the answer letter (strict parser
39.23 vs first-letter 77.81 on the same checkpoint).

## Measurement discipline

Every arm reports recover, broke, the rotated-choice control, and strict-vs-first-letter
parsing. A/A noise floor is 0.50 points. Coverage is measured functionally — put the
window in context and see whether the model answers — not by string overlap, which gave a
meaningless 20.5%.

Contamination was re-checked: 2,937 test questions appear verbatim inside the kit, because
the source windows contain the same sentences GPT-3.5 used to write them. The gain is equal
on both groups (+7.4 with overlap, +7.1 without), and no copy of the dataset was found in
the corpus.

Label noise in the benchmark is estimated at 3–5% by a 31B judge reading source windows,
putting the ceiling with perfect knowledge at roughly 95–97. The `broke` population is
about twice as rich in ambiguous labels as a random sample, so roughly a third of `broke`
is unavoidable when the model follows the source.

## Current run and outlook

**Big run 3** is packing now: the full kit re-tokenised at block 2048 so that ~1.36k-token
windows are no longer split, 2.57B tokens including recall-QA ×3 and sibling ×2, on a new
FSDP2 + Liger stack measured at 78.6k tok/s in smoke test (twice the previous DDP stack),
0.5M tokens per step. Estimated 10–11 hours, followed by one consistency pass.

Realistic expectation is **80–81**. Reaching 85 requires 531 more questions:

| Group | ~n | State |
|---|---|---|
| Source available, unfixed, gold already ranked second | 660 | partially learned — needs better extraction, not more data |
| No retrievable source | 520 | corpus ceiling; needs deeper IEEE and research coverage |
| `broke` from mislearning | 330 | wider sibling-contrast plus consistency |
| `broke` from ambiguous labels | 115 | not fixable without using the answer key |

Untried levers, each estimated at ≤ +1: prompt distillation (aimed directly at the
extraction bottleneck), window quality filtering before generation, a 122B teacher for the
uncovered group, and mixing think-mode rows into training.
