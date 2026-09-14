# teleqna — specialist track

The largest column on the GSMA Open Telco leaderboard: 10,000 of the 19,588
questions the `average` is computed over, more than the other six benchmarks
combined. This directory holds everything for that track; nothing here touches
the TeleLogs or srsranbench pipelines.

## The benchmark

| | |
|---|---|
| Config | `GSMA/ot-full`, config `teleqna`, split `test` |
| Local parquet | `data/teleqna/test-00000-of-00001.parquet` |
| sha256 | `73f3cd724d8ff3c1071fbb135e80d69a54b0773acde590006949a67956338e57` (= the HF LFS oid, byte-identical) |
| Readable mirror | `data/teleqna/test.jsonl`, regenerate with `profile_data.py` |
| Fields | `question: str`, `choices: list[str]` (2–5), `answer: int` (0-based), `subject: str` |
| Rows | 10,000 |
| Origin | [TeleQnA](https://arxiv.org/abs/2310.15051), Maatouk et al. 2023 (Huawei Paris / Yale) |

Five subjects, and they are not equally hard:

| subject | rows | what it tests |
|---|---:|---|
| Research publications | 4,500 | claims from transactions and conference papers |
| Research overview | 2,000 | broad telecom research topics |
| Standards specifications | 2,000 | 3GPP/IEEE clause-level detail |
| Standards overview | 1,000 | standards summaries |
| Lexicon | 500 | terminology and definitions |

1,810 questions carry an explicit `[3GPP Release NN]` tag (mostly 17 and 18),
and many more carry `[IEEE ...]` or `[TCP/IP]` markers. That tag is a free
routing feature: it says which corpus the answer lives in.

## The fact that defines this track: the benchmark is fully public

The GSMA `teleqna` split is the public TeleQnA dataset verbatim. Verified:

- 9,948 / 9,948 distinct question strings appear word-for-word in
  [`netop-team/TeleQnA`](https://github.com/netop-team/TeleQnA);
- 9,951 / 10,000 rows match on option text, option **order**, and gold index;
- row *i* of the parquet is `question i` of the public file.

The public copy additionally ships an **`explanation` field for all 10,000
questions** — the reasoning behind each gold answer, written by the generation
pipeline and human-checked. The authors password-protected the zip
(`teleqnadataset`) specifically "to prevent inadvertent data contamination with
models trained using GitHub data". `GSMA/ot-lite`'s `test_teleqna.json` (a
1,000-row 10% stratified sample, all of it inside ot-full) ships the same field
un-protected, under the key `explaination`.

Two consequences, and they pull in opposite directions:

1. **Contamination is the null hypothesis, not the exception.** The dataset has
   been on GitHub since 2023 and on HuggingFace (gated) since. Any leaderboard
   score on this column is a mix of telecom knowledge and memorisation, and
   there is no way to separate them from the score alone. Before believing any
   gain, this track has to run a memorisation probe (see *Open decisions*).
2. **Nothing prevents fine-tuning directly on the test set.** A 10,000-row
   question→gold→explanation file is exactly an SFT corpus. Doing that would
   score very high and mean nothing. It is the same category error as the
   TeleLogs `v2` mute classifier this repo already documented, one order of
   magnitude larger. **This track does not train on `data/teleqna/`.**

## The official harness contract

[`gsma-labs/evals`](https://github.com/gsma-labs/evals/blob/main/src/evals/teleqna/teleqna.py),
Inspect AI — identical in shape to srsranbench:

```python
Sample(input=record["question"], choices=record["choices"],
       target=chr(65 + record["answer"]), metadata={"subject": record.get("subject")})
Task(dataset=..., solver=multiple_choice(cot=False), scorer=choice())
```

- `cot=False` selects a prompt, it does not forbid reasoning. It swaps
  "the entire content of your response" for "the last line" and drops "Think
  step by step". `parse_answers()` keeps the **last** `ANSWER:` match, so a
  model may reason first. The real constraints are operational: a well-aligned
  model reading "the entire content" drops CoT on its own, and reasoning that
  runs past `max_tokens` never emits the line and scores wrong.
- **Choices are never shuffled.** `multiple_choice()` defaults `shuffle=False`.
- Variable choice count: 2 choices (16 rows), 3 (87), 4 (3,456), 5 (6,441). The
  `{letters}` list in the prompt is per-row, so `E` only exists on 5-choice rows.

## What the data actually looks like (`profile_data.py`)

```
rows                     10000     choices/row      2:16  3:87  4:3456  5:6441
question chars   p50 81 / p99 172   choice chars   p50 36 / p99 136
```

### There is no position prior — unlike srsranbench

| gold index | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| count | 2210 | 2179 | 2169 | 2153 | 1289 |
| share | 22.10% | 21.79% | 21.69% | 21.53% | 12.89% |

`always A` scores **22.10%** against a 21.89% random baseline. The E share is
low only because just 6,441 rows have an E; within 5-choice rows E is 20.0%.
The TeleQnA generation pipeline explicitly shuffles options, and it worked.

This is the opposite of srsranbench, where `always A` scores 75.83%. **The
srsranbench playbook does not transfer.** Per-subject best fixed letter never
exceeds 24.2% (Lexicon, C).

### The prior that does exist: distractor shape

| pattern | rows containing it | rows where it is gold | gold rate |
|---|---:|---:|---:|
| `All of the above…` | 824 | 740 | **89.81%** |
| `Both … and …` | 145 | 99 | **68.28%** |
| `None of the above…` | 647 | 2 | **0.31%** |
| `not mentioned/specified…` | 17 | 0 | **0.00%** |

Plus a length prior: the gold choice averages 43.8 characters against 39.9 for
distractors, and is the longest option in 3,722 rows against shortest in 2,550.

Stacked into a policy that never reads the question — take "All of the above" if
present, else "Both …", else drop "None of the above" and take the longest
survivor — this scores **39.72%** (`run_baseline.py --heuristic`). Permuting the
choices barely moves it (39.42%), confirming it is a content-shape artefact and
not a position one.

**39.72%, not 22%, is the floor.** Any reported gain has to clear it.

### Data hygiene

- 9,944 unique question strings; 56 rows repeat an earlier question.
- 3 rows are exact duplicates (question + choices) of another row.
- 0 keys carry contradictory gold answers — cleaner than srsranbench, which had 12.

So the ceiling is effectively 100% and a stem-level dev split is safe, if one
were ever carved (see the contamination warning above — it should not be).

## Prior work and where SOTA actually is

TeleQnA is the *founding* benchmark of the telecom-LLM literature and everything
below cites it. Unlike srsRANBench, it has genuine independent competition.

**The original paper** ([arXiv:2310.15051](https://arxiv.org/abs/2310.15051))
established the reference points, on the same 10,000 rows:

| | overall | Lexicon | Res. overview | Res. publications | Std. overview | Std. specs |
|---|---:|---:|---:|---:|---:|---:|
| GPT-4 | 74.91% | 86.80% | 76.25% | 77.62% | 74.40% | 64.78% |
| GPT-3.5 | 67.29% | 82.20% | 68.50% | 70.42% | 64.00% | 56.97% |
| **Active telecom professionals** | **64.86%** | 80.33% | 63.66% | 68.33% | 61.66% | 56.33% |

Two things to carry forward. First, GPT-4 already beat human telecom
professionals in 2023 — this benchmark stopped being a human-parity target long
ago. Second, the difficulty ordering is stable across every model since:
**Lexicon easiest, Standards specifications hardest**, ~22 points apart. A
specialist earns its points in *Standards specifications* (2,000 rows) and
*Research publications* (4,500 rows), or it earns nothing.

**The approaches that have been tried**, in the order they matter here:

| line of work | method | reported |
|---|---|---|
| [Telco-RAG](https://arxiv.org/abs/2404.15939) (2024) | RAG over 3GPP Rel-18 docs, tuned chunking/embedding/indexing, dual-round retrieval | 75.4% on their 3GPP subset, ~+20 pts over GPT-3.5 |
| [Telco-oRAG](https://arxiv.org/abs/2505.11856) (2025) | + glossary-enhanced query refinement, neural 3GPP-series router, parallel web retrieval | 76.2%; **90.8% on the Lexicon subset** vs 80.2% no-context; MCQ 84.7% with GPT-OSS-120B vs 68.4% baseline (250 questions) |
| [Tele-LLMs](https://arxiv.org/abs/2409.05314) (2024) | continual pretraining of 1B–8B models on Tele-Data, 5,000 A6000-hours | evaluates on their own Tele-Eval, **not** on TeleQnA; ~25% relative gain there |
| [TelecomGPT](https://arxiv.org/abs/2407.09424) (2024) | full telecom SFT pipeline | reports TeleQnA among several tasks |
| [ORANSight-2.0](https://arxiv.org/abs/2503.05200) (2025) | RANSTRUCT: RAG-teacher generates instructions, QLoRA fine-tune | the srsranbench track's reference; O-RAN focus |
| [AT&T OTel 2.0](https://github.com/farbodtavakkoli/OTel) (2026) | full-parameter post-training of Gemma-4-31B-IT on ~400B telecom tokens; also embedding + reranker + abstention models | leaderboard #1 |

The consistent finding across all of them: **retrieval is worth 10–20 points on
the standards half and almost nothing on the research half.** Telco-oRAG's own
ablation is the cleanest statement of it — +10.6 points on Lexicon from context,
and the entire Telco-RAG line targets 3GPP documents specifically.

**The GSMA leaderboard, and the same scale problem as srsranbench.** Inverting
the published binomial stderr on the `teleqna` column gives the sample count:
**32 of 85 models sit at n ≈ 10,000** and **53 at n ≈ 1,000**. The column mixes
`ot-full` and `ot-lite` runs in a single ranking. Unlike srsranbench, the
constant is the same (~22%) on both, so the mixing distorts precision rather
than the baseline — but it still means a 0.845 lite row and a 0.912 full row are
not comparable.

| model | teleqna | n≈ | note |
|---|---:|---:|---|
| OTel-LLM-8.3B-QnA (AT&T) | **0.9120** | 10,237 | leaderboard #1 overall; the only >0.86 at full scale |
| gemini-3.1-pro-preview | 0.8520 | 1,005 | best frontier model, lite scale |
| claude-opus-4.5 / 4.6 | 0.845 / 0.844 | ~1,000 | |
| gpt-5 | 0.8380 | 999 | |
| LTM (SoftBank) | 0.8191 | 11,799 | second-best full-scale specialist |
| gpt-5.2 / gpt-5.1 | 0.8270 | 999 | |
| gpt-5-mini | 0.4470 | 999 | **outlier — see below** |
| claude-sonnet-4.6 | 0.4480 | 1,003 | **outlier** |
| mixtral-8x7b | 0.4878 | 9,994 | |
| falcon3-7b | 0.1612 | 9,877 | **below the 21.9% random baseline** |

The three flagged rows are almost certainly harness failures, not knowledge
failures: `gpt-5-mini` at 0.447 sits 39 points below `gpt-5`, `claude-sonnet-4.6`
at 0.448 sits 37 points below `claude-sonnet-4.5`, and a score *below random*
is only reachable by systematically failing to emit a parsable `ANSWER:` line.
This is the same truncated-reasoning failure that cost this repo 299/320
samples in commit `09fa2ed`. It is a warning about the submission, not the model.

**What this sets as the target.** The honest bracket:

```
21.89%  random
39.72%  distractor-shape heuristic, no telecom knowledge   <- the real floor
64.86%  active telecom professionals
74.91%  GPT-4 (2023)
83.80%  gpt-5 (n≈1000)
91.20%  AT&T OTel-LLM-8.3B-QnA (n≈10237)                   <- SOTA
```

An 8B specialist beating 0.912 at full scale would be leaderboard #1 on this
column. An 8B specialist at 0.85 would already sit above every frontier model
that has been run — but only if the run is at n=10,000, and only if it is not
memorising a public test set.

## Measured: Qwen3-8B closed-book baseline

All 10,000 rows, greedy, 32 max tokens, no native thinking, Inspect template
verbatim. 0 truncations, 1 unparsable response out of 10,000.

| arm | overall | Lexicon | Res. ovw | Res. pub | Std. ovw | Std. spec | tok/case |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-think | 71.95% | 83.40% | 74.25% | 74.84% | 69.70% | 61.40% | 5.0 |
| no-think, rotated | 70.36% | 83.60% | 73.50% | 73.07% | 69.80% | 58.10% | 5.0 |
| **thinking** | **73.97%** | 87.20% | 75.55% | 77.62% | 72.20% | **61.75%** | 967.8 |
| thinking, rotated | 73.58% | 87.60% | 75.70% | 76.38% | 73.50% | 61.70% | 1000.2 |
| GPT-4 (2023 paper) | 74.91% | 86.80% | 76.25% | 77.62% | 74.40% | 64.78% | — |
| Active telecom professionals | 64.86% | 80.33% | 63.66% | 68.33% | 61.66% | 56.33% | — |
| distractor-shape heuristic | 39.72% | 32.80% | 39.90% | 41.80% | 42.00% | 35.50% | 0 |

Truncation was 0 / 0 / 7 / 10 out of 10,000 — the 6,000-token cap was adequate
and did not manufacture the failure mode that sank the sub-random leaderboard
rows.

### Thinking: +2.02 points, and it buys order-robustness

Paired McNemar on all 10,000 rows:

| comparison | Δ | discordant | p |
|---|---:|---:|---:|
| thinking vs no-think | **+2.02** | 554 / 756 | <0.001 |
| no-think: rotated vs plain | −1.59 | 647 / 488 | <0.001 |
| **thinking: rotated vs plain** | **−0.39** | 538 / 499 | **0.238** |

The second effect matters as much as the first. Under no-think, rotating the
choices costs a significant 1.59 points; under thinking it costs nothing
measurable (p = 0.24). Reasoning does not just raise the score, it removes the
presentation sensitivity — the model stops being nudged by where an option sits.

**This is the opposite of srsranbench**, where thinking *cost* 4.8 points
(77.70% vs 82.49%). The two benchmarks reward different things: srsRAN
questions are identifier lookups where deliberation only adds noise; teleqna
has 3,000 standards rows that need multi-step reading of a clause.

### Where thinking does and does not help

| subject | rows | no-think → thinking |
|---|---:|---:|
| Lexicon | 500 | +3.80 |
| Research publications | 4,500 | +2.78 |
| Standards overview | 1,000 | +2.50 |
| Research overview | 2,000 | +1.30 |
| **Standards specifications** | 2,000 | **+0.35** |

Thinking pays everywhere except the single hardest subject, where it does
essentially nothing. That is diagnostic: **Standards specifications is a
knowledge gap, not a reasoning gap.** No amount of deliberation reconstructs a
3GPP clause the model never read. Those 2,000 rows are exactly where retrieval —
or knowledge distilled from the specs — is the only lever, and where 766 rows
remain wrong.

Note also that the thinking arm's Research publications score is 77.62%, the
same number to two decimals as GPT-4's in the 2023 paper, and the overall gap to
GPT-4 closes from 2.96 to 0.94 points.

Three readings of the no-think arm, all still standing:

- **The score is knowledge, not artefact.** +32.2 points over the shape
  heuristic, and the predicted-letter histogram is near-uniform
  (A 2060 / B 2280 / C 2368 / D 2160 / E 1131, and only 6,441 rows have an E).
- **The subject profile reproduces the 2023 paper exactly** — Lexicon easiest,
  Standards specifications hardest, 22 points apart, same ordering GPT-4 and
  GPT-3.5 showed.
- **The gap to GPT-4 is 2.96 points and it sits in the standards half.** By
  rows: Research publications 125, Standards specifications 68, Standards
  overview 47, Research overview 40, Lexicon 17. Standards specifications is
  also the only subject that loses meaningfully under rotation (−3.3 points
  against ≤1.8 everywhere else), i.e. the subject where the model is guessing
  rather than knowing. That is exactly where Telco-oRAG says retrieval pays.

Arithmetic on the target: reaching AT&T's 0.912 needs 1,925 more correct rows
out of the 2,805 currently wrong — 69% of every remaining error. A realistic
first-round target is 83–85%, which would clear every frontier model on the
board at lite scale, and needs 40–47% of the errors fixed.

## Measured: memorisation probe

600-row stratified sample (270/120/120/60/30, proportional), every arm paired
row-for-row, McNemar with an exact two-sided binomial on the discordant pairs.

| arm | accuracy | Δ vs control | discordant | p |
|---|---:|---:|---:|---:|
| `orig` | 72.67% | — | — | — |
| `perm` | 69.00% | −3.67 | 55 / 33 | 0.025 |
| `perm`, all 10,000 rows | 70.36% | **−1.59** | 647 / 488 | <0.001 |
| `distract` | 88.50% | **+15.83** | 50 / 145 | <0.001 |
| `paraphrase` (576 kept) | 72.57% | **−0.87** | 25 / 20 | **0.551** |

- **`paraphrase` is the load-bearing result.** Rewriting every question in
  different words, with all technical terms, acronyms and `[Release NN]` tags
  preserved, costs nothing measurable (p = 0.55). A model recalling question
  strings collapses here. It does not.
- **`perm` is real but small.** What is more informative than the −1.59 net is
  that **1,135 of 10,000 rows flip** under rotation (647 lost, 488 gained). The
  model is genuinely undecided on ~11% of the benchmark; only the 159-row
  asymmetry is a position effect.
- **`distract` (+15.83) says ~16 points of this benchmark's difficulty live in
  the distractors, not the questions.** TeleQnA's distractors are close in
  meaning to the gold. Consequence for any generated training set: a teacher
  that writes lazy, off-topic distractors will teach the answer without
  teaching the discrimination the benchmark actually scores. Distractor
  generation is a first-class step, not a detail.

### TS-Guessing, and a control that had to be thrown away

Masking one **wrong** option and asking the model to reproduce it gave
**29/599 = 4.84%** exact match — inside the 2–10% band the pre-registered
`verdict()` calls partial exposure. The length breakdown then falsified the
assumption that threshold rested on:

| masked option length | n | distractor EM | gold EM |
|---|---:|---:|---:|
| 1 word | 63 | **15.87%** | 36.67% |
| 2–3 words | 155 | 7.74% | 18.70% |
| 4–6 words | 145 | 2.07% | 3.01% |
| 7–12 words | 194 | **1.55%** | 4.73% |
| 13+ words | 42 | 2.38% | 6.17% |
| **≥5 words** | 326 | **1.23%** | 4.50% |

Monotone decay with length is the signature of guessing. Recall would be flat
or rising, since longer strings are more distinctive. 75.9% of the exact matches
are ≤3 words (`Touch`, `Tensorflow`, `7`, `Sigmoid`, `Class B`) against 34.4%
of the misses.

That is an argument, not a measurement, so a control was run — and **it was not
a valid one**, which is recorded here rather than quietly dropped. Masking a
distractor transplanted from a different question scored **0/599**, mean token
F1 0.032 against 0.299 on the real probe. But a transplanted distractor is not a
plausible answer to the question it was pasted into, so that control removed
guessability along with memorisation. It measured "can the model reproduce
unrelated text", which is trivially no.

The valid control needs text that is plausible for *this* question and
impossible to have been memorised. Only one source qualifies: distractors the
served model writes fresh, length-matched, at run time — `make_selfgen_control.py`.
Its rate is an upper bound on guessability, because a model reproducing its own
recent output has an advantage no third-party text gives it. Compared per length
band against the real probe:

- real ≤ control → the observed matches need no memorisation to explain
- real > control → the excess is what recall has to account for

**Result: real ≤ control.**

| masked option length | real probe (TeleQnA distractor) | control (self-generated) | excess |
|---|---:|---:|---:|
| 1 word | 10/63 = 15.87% | 4/47 = 8.51% | +7.36pp |
| 2–3 words | 12/155 = 7.74% | 7/136 = 5.15% | +2.59pp |
| 4–6 words | 3/145 = 2.07% | 3/145 = 2.07% | 0.00pp |
| 7–12 words | 3/194 = 1.55% | 1/183 = 0.55% | +1.00pp |
| 13+ words | 1/42 = 2.38% | 2/36 = 5.56% | −3.17pp |
| **all** | 29/599 = 4.84% | 17/547 = 3.11% | +1.73pp (z = 1.51) |
| **≥5 words** | 4/326 = **1.23%** | 5/300 = **1.67%** | **−0.44pp** (z = −0.46) |

Mean token F1 is 0.299 on the real probe and 0.282 on the control — the model
reconstructs a genuine TeleQnA distractor no better than one invented seconds
earlier. On options long enough that guessing is hard, it does slightly *worse*.
The whole raw difference sits in the one-word band, where the answer space is
small enough that the comparison carries almost no information, and it is not
significant overall.

### Verdict

**No usable evidence that Qwen3-8B has memorised TeleQnA.** Four independent
lines agree:

1. TS-Guessing does not exceed a measured guessing floor at any length where
   guessing is hard.
2. Paraphrasing every question costs nothing (p = 0.55).
3. Rotating the choices costs 1.59 points on 10,000 rows — a position effect,
   not an item-recall effect.
4. The per-subject difficulty profile reproduces the 2023 paper's ordering and
   spread, i.e. the model is failing where the questions are hard rather than
   where its exposure was thin.

This is not proof the model never saw the file. It is the stronger and more
useful statement that **71.95% is not explained by recall**, so the gate is
passed and the numbers from this track can be built on. The probe should be
re-run against any fine-tuned checkpoint, since training is where contamination
would newly enter.

## Measured: prompt optimisation, and its ceiling

DSPy 3.2.1 / GEPA, reflective prompt evolution. The optimiser rewrites a
**system prompt**; the harness's user message passes through byte-identical, so
the artefact is deployable as-is behind our own endpoint. (The user message is
*not* fixed by any rule — we serve the model — but holding it fixed here makes
this a clean control for the structural changes that come next.) Splits:
400 train / 600 val / 9,000 held-out, stratified by subject, disjoint.
Every reported number comes from `run_baseline.py --system`, i.e. the same
scorer, template and decoding as the baseline.

### It does not work, and the loss is significant

| comparison | Δ pp | discordant | p |
|---|---:|---:|---:|
| no system → seed prompt | −0.34 | 413 / 382 | 0.287 |
| no system → GEPA prompt | **−1.12** | 524 / 423 | **0.00114** |
| seed → GEPA | **−0.78** | 315 / 245 | **0.00351** |

GEPA scored +1.00 pp on val and **−0.78 pp** on held-out. Complete reversal.

The loss is substantive, not a format break — `summary.json` for all three arms
reports parse failures ≤ 2 and truncations ≤ 2 out of 9,000, and mean completion
length is 5.0 tokens everywhere. The model still emits `ANSWER: X`; it picks the
wrong letter.

Three details name the mechanism:

* **Knowledge wins fell.** Rows the shape heuristic gets wrong and the model
  gets right: 3,608 → 3,575 (−33). The prompt did not add knowledge; it
  displaced some.
* **The damage concentrates on shape-cued rows** — −3.92 pp on the 868 rows
  containing "All of the above"/"Both …", against −0.82 pp on the other 8,132.
  Feeding the model scattered facts disrupts exactly the heuristic it had.
* **The one real gain is not GEPA's.** Lexicon +4.22 pp, of which seed→GEPA
  contributes +0.00. A single framing sentence is worth four points on Lexicon
  and costs points on every other subject.

GEPA's evolved prompt is a cheat sheet, not a strategy — it literally contains
`"IEEE 802.11 SME = management protocols, not PLME"` and `"ARPU = Average
Revenue Per User"`, verbatim dev-set answers.

### The ceiling: fit on all 10,000, score on the same 10,000

Not a result and not submittable. It is the largest number prompt-only fitting
can produce when the optimiser is handed the answer key, and it exists to set
the discount on every future prompt-tuning number here.

| | fit on | scored on | Δ vs no-system |
|---|---|---|---:|
| honest | 400 + 600 | disjoint 9,000 | −1.12 pp |
| ceiling | all 10,000 | the same 10,000 | **−0.45 pp** |
| | | **leakage = ceiling − honest** | **+0.67 pp** |

**With total leakage the prompt is still worse than no prompt at all.** Even
in-sample, on rows it was directly optimising with gold answers visible, GEPA
moved val from 71.35% to 71.60% — +0.25 pp.

The ceiling run reproduces every sign of the honest run on 10,000 rows:
Standards specifications −2.80 pp (p = 0.00053), shape-cued rows −3.01 pp
(p = 0.0019), knowledge wins −18, Lexicon +4.60 pp (p = 0.0011).

### What this settles

**The system prompt is not a lever on this benchmark, and the failure is not a
tuning failure — there is no headroom to find.** The measured leakage budget is
0.67 pp, so no future prompt result on this track is worth more than that even
before it is discounted.

This is coherent with the baseline. Rotating the choices flips 1,135 of 10,000
rows: about 11% of the benchmark sits on a knife edge. Any perturbation — a
rotation, a system message — churns that band and nets ≈ 0, because it wins and
loses at nearly equal rates (413/382 for the seed prompt). GEPA does worse than
noise because its injected facts are specific and mostly irrelevant to the
question at hand, so they actively pull. And it matches the thinking result:
reasoning tokens are strictly more powerful than a system prompt and bought only
+2.02 pp overall, +0.35 pp on Standards specifications.

Standards specifications is a knowledge gap. It does not move for prompting, it
does not move for deliberation, and it will not move until the facts are put
somewhere the model can reach.

## Measured: the provenance of the questions

The `[...]` tag at the end of a question is the only provenance marker the
benchmark carries, and it is not evenly distributed:

| subject | rows | tagged | 3GPP-tagged |
|---|---:|---:|---:|
| Standards specifications | 2,000 | **2,000** | 1,509 |
| Standards overview | 1,000 | 618 | 301 |
| Research publications | 4,500 | 3 | 0 |
| Research overview | 2,000 | 0 | 0 |
| Lexicon | 500 | 0 | 0 |

**A 3GPP corpus can reach at most 1,810 rows — 18% of the benchmark.** Any plan
that leads with 3GPP retrieval is optimising the smallest addressable slice.
The literature reads the other way only because 3GPP is the slice with a
downloadable corpus.

Standards specifications partitions exactly, no row unaccounted for:
Rel-18 780, Rel-17 641, IEEE 802.11 298, IEEE C95.1 130, IEEE 802.15.4 63,
Rel-14 55, Rel-19 17, Rel-16 16. Rel-17+18 alone is 1,421 rows (71%).
**IEEE C95.1 is 130 test rows out of a single document** — the highest
leverage-per-page anywhere in the benchmark.

Routing below release level is not possible from the question text: only 13 of
10,000 rows cite a TS/TR number. The one narrow exception is 34 rows carrying a
5G SBI service-operation name (`Nmfaf` 8, `Ndccf` 8, `Nadrf` 7, `Nnef` 4,
`Nnwdaf` 3), which point at TS 23.288 / 29.5xx.

### One generator made four of the five subjects

| subject | mean options | "All of the above" | "None of the above" |
|---|---:|---:|---:|
| Standards specifications | 4.61 | 114 (5.7%) | 91 |
| Standards overview | 4.63 | 72 (7.2%) | 53 |
| Research publications | 4.62 | 423 (9.4%) | 334 |
| Research overview | 4.66 | 212 (10.6%) | 169 |
| **Lexicon** | **4.73** | **3 (0.6%)** | **0** |

The first four match too closely to be coincidence; Lexicon does not, and was
built from a dictionary by a different path. **This is where the 39.72% shape
prior comes from** — it is a fingerprint of the GPT-3.5 generator, not a
property of telecommunications. Any regenerated training set that does not
reproduce this distribution will teach a heuristic that does not match the test.

### The generation pipeline, from the paper

~25,000 pages / ~6M words of source. Two GPT-3.5 agents: a generator producing
MCQs with a cited supporting sentence, and a validator answering blind to
confirm. Telecom experts reviewed every question for accuracy and
self-containment; K-Means over Ada v2 embeddings removed near-duplicates.
Research overview came from surveys selected by 15-year citation count plus
practitioner recommendation; Research publications from technical articles and
open-access books across many publishers.

### `explanation`: obtained for all 10,000, and what it is not

`TeleQnA.zip` from `netop-team/TeleQnA` is AES-encrypted; `unzip` fails with
"unsupported compression method 99" and needs `pyzipper`. Password
`teleqnadataset`, as published. Alignment against our parquet is exact:
**10,000/10,000 question strings match by index, 10,000/10,000 gold indices
agree** — stronger than the string-set check recorded above.

It does **not** carry provenance. Median 19–20 words, and across 10,000 rows:
27 mention a TS/TR number, 1 contains "et al.", 2 contain a quoted title.
It restates the supporting fact, it does not cite where the fact came from.
**The source documents behind the 7,000 research and lexicon rows remain
unidentified, and the explanation field does not identify them.**

What it does give is the exact fact each row tests, for every row — which makes
one experiment possible that was not before: score the benchmark with the gold
explanation injected as context. That is the ceiling of any retrieval system on
this benchmark, and it should be measured before a corpus is built, not after.
Note it must not be used to *select* corpus documents; that is test-set-informed
selection and leaks. Coverage audit only, reported and not optimised against.

## Measured: AdaptKey-Nemotron-30b is trained on this test set

`AdaptKey/AdaptKey-Nemotron-30b` (LoRA-tagged but shipped as 13 full shards on
`nvidia/Nemotron-3-Nano-30B-A3B`) publishes its training corpus. It reports a
"GSMA Open-Telco Composite Score 596 vs baseline 538", self-reported,
`verified: false`.

Scanned all 1,303,277 records of `training_data/train.jsonl` (5.08 GB) by 8-gram
shingle against every TeleQnA question and gold option:

| | |
|---|---:|
| records | 1,303,277 |
| **TeleQnA test questions found** | **8,674 / 10,000** |
| TeleQnA gold option texts found | 3,577 |
| contaminated records | 11,208 (0.86%) |
| records in the MCQ shell | 13,474 |
| distinct output prefixes | 234,320 (18%) |
| median output length | 7 words |

**The 1,326 "misses" are the detector's blind spot, not absences.** An 8-gram
shingle cannot exist for a question shorter than 8 words, and 1,312 questions are.
The two distributions line up almost exactly:

| subject | missed | < 8 words | diff |
|---|---:|---:|---:|
| Lexicon | 305 | 305 | **0** |
| Research publications | 520 | 512 | 8 |
| Research overview | 198 | 196 | 2 |
| Standards overview | 127 | 126 | 1 |
| Standards specifications | 176 | 173 | 3 |
| **total** | **1,326** | **1,312** | **14** |

Two short questions were checked by hand and are both in the corpus with the gold
answer as the target — `#3282` "What is electromotive force (emf)?" and `#9783`
"What is the capacity of a channel?". So the corpus holds on the order of
**9,986 of the 10,000 rows**: effectively the entire benchmark.

Not paraphrase. Verbatim, with the full option list, with TeleQnA's own category
label, and with the gold answer as the SFT target:

```
input:  • context: category: standards specifications. this is a
        telecommunications standards question.
        • question: what do the angles bearing, downtilt, and slant represent?
        options: 1. rotation angles of the lcs with respect to the gcs
                 2. unit vectors of the gcs  3. polarized field components …
output: **Answer:** Rotation angles of the LCS with respect to the GCS
```

**Consequences, in order of importance:**

1. **The corpus is unusable here.** Training on it is training on our test set.
   This is a blocking finding, not a caveat.
2. **Its leaderboard number carries no information about this column**, and the
   published weights are contaminated by construction.
3. **Deleting the contaminated 0.86% does not leave a usable corpus**, because
   the contaminated part *is* the only knowledge QA in it. Classifying 140,000
   records sampled across seven offsets:

   | share | content |
   |---:|---|
   | 58.3% | KPI statistics prose — "The variance of PRB_Utilization_UL is 0.374" |
   | 16.1% | KPI trend — "The trend of PRBs_DL_Current is 0" |
   | 15.5% | open5gs NetworkSlice / AMF YAML |
   | 8.9% | ultra-short JSON slice profiles |
   | 1.1% | the MCQ shell — i.e. TeleQnA |
   | 0.1% | `\boxed{C1..C8}` gate labels — the TeleLogs schema |

   Roughly 99% is synthetic telemetry arithmetic and config templating for the
   TeleYaml / TeleLogs / srsRAN columns. Zero records anywhere in `train.jsonl`
   cite a 3GPP source document. There is no telecom *knowledge* in this corpus
   other than the test set itself.
4. The only spec-grounded slice is 500 records in `test.jsonl`, quoting TS 38.101
   / 38.104 / 38.211-213 RF and PHY tables. TeleQnA's standards rows are Rel-17/18
   service-and-system-aspects plus IEEE 802.11 / C95.1. Almost disjoint.

### It is not only this track

Scanned the same corpus against the other two test sets in this repo, with the
shared C1–C8 boilerplate header stripped:

| test set | matched | |
|---|---:|---|
| teleqna (10,000) | **~9,986** | 99.9% |
| **TeleLogs official 864** | **864 / 864** | **100%** |
| srsRANBench (1,502) | 1 / 1,502 | 0.1%, clean |

The TeleLogs hits are not boilerplate. `setdefault` assigns each shingle to the
first item that produced it, so text shared across all 864 items can only ever
credit one of them — reaching 864/864 requires each item's *own* content to be
present. It is: the matches land on per-item drive-test rows
(`2025-05-07 11:03:33.000000|128.148154|32.624705|11|745|-85.82|…`) and per-item
AAU engineering-parameter tables.

So AdaptKey's "GSMA Open-Telco Composite 596 vs 538" is memorisation on at least
two of the columns it averages. srsRANBench is the one place it is clean, and
that is also the one place its corpus has no relevant content.

**The lesson: shingle-scan every third-party corpus and checkpoint against every
test set we own, before use.** `scan_contamination.py` does it in one streaming
pass. Note its blind spot — items shorter than `k` words are invisible — so
treat every count it prints as a lower bound.

## Measured: the corpus exists, and the OTel training set is clean

The teleqna column of `GSMA/leaderboard`, all 88 rows:

| model | provider | teleqna | average |
|---|---|---:|---:|
| OTel-2.0-LLM-31B-IT | AT&T | **0.917** | 0.903 |
| OTel-LLM-8.3B-QnA | AT&T | **0.912** | 0.860 |
| TeleLLM | China Telecom | 0.894 | 0.758 |
| gemini-3.1-pro-preview | Google | 0.852 | 0.756 |
| gpt-5 | OpenAI | 0.838 | 0.719 |
| qwen3-235b-a22b-2507 | Qwen | 0.797 | 0.636 |
| qwen3-8b | Qwen | 0.745 | 0.411 |

Median over the 88 is 0.766. Only two rows clear 0.90 and both are AT&T. Our own
closed-book Qwen3-8B measured 0.7195 against the board's 0.745 — close enough to
confirm the harness is faithful, the residual is think-mode and sampling.

The comparison that decides the track:

```
qwen3-8b           (8B,   generic)   0.745
qwen3-235b-a22b    (235B, generic)   0.797    +5.2pp for 30x the parameters
OTel-LLM-8.3B-QnA  (8.3B, domain)    0.912   +16.7pp at the same size
```

Scaling buys 5 points, domain post-training buys 17, and an 8.3B specialist beats
every frontier model on the board. Base-model swap is not the lever; it was ranked
first here before this was measured and that ranking was wrong.

### The corpus is published, ungated

`TelecomGPT-R1`'s card claims "1.0 on TeleQnA". That is normalised radar scaling
and the model is not in the leaderboard snapshot at all. Trust the parquet.

What is real is that GSMA published the training corpora behind OTel:

| repo | on disk | contents |
|---|---:|---|
| `GSMA/3GPP` | 3.14 GB | 15,052 spec markdown files, Rel-8 → Rel-20 |
| `GSMA/Telco-Common-Corpus` | 8.9 GB | 1.78M docs, ~10B tokens — 64k IEEE open-access articles, 10k RFCs + 40k drafts, 120k patents, OpenAlex, Wikidata |
| `farbodtavakkoli/OTel-LLM` | 1.4 GB | 606,237-record SFT set, apache-2.0 |
| `GSMA/etsi` `itu` `oran` `gsma` `camara` `tmforum` | — | the other six SDOs |
| `GSMA/vector_databases` | — | prebuilt Chroma indices per SDO |
| `farbodtavakkoli/OTel-Embedding` / `-Reranker` | 22M–8B | domain-tuned retrieval stack |

`GSMA/3GPP` has 84,220 files; `snapshot_download` fails on it (connection reset in
the metadata phase, local dir never created). `GIT_LFS_SKIP_SMUDGE=1 git clone
--depth 1` works — one connection, LFS images left as pointers.

Coverage against the 10,000, using the provenance tags counted above:

| slice | rows | source | |
|---|---:|---|:--:|
| Research publications + overview | 6,500 | 64k IEEE open-access articles + OpenAlex | ✅ |
| 3GPP Rel-14→19 | 1,810 | `GSMA/3GPP`, every release present and over-covered | ✅ |
| Lexicon | 500 | Wikipedia/Wikidata slice | ✅ thin |
| Standards overview, untagged | 382 | mixed | ⚠️ |
| TCP/IP | 118 | 10k RFCs + 40k drafts | ✅ |
| ETSI NFV Rel-5 | 38 | `GSMA/etsi` | ✅ |
| IEEE 802.11 / 802.3 / 802.15.4 | 517 | not in any GSMA repo | ❌ |
| IEEE C95.1 | 130 | paid IEEE standard | ❌ |
| Bluetooth | 5 | not present | ❌ |

**9,348 of 10,000 (93.5%) have a downloadable source document.** The gap is 652
rows of IEEE and Bluetooth standards. The 802 family is free through the IEEE GET
program; C95.1 is not. This closes the open question from the provenance section —
the 6,500 research rows whose sources "remain unidentified" are IEEE open access.

### Scanning it: 8-gram flags 530, real leakage is 15

`scan_contamination.py` on `OTel-LLM-Data.jsonl`, fields `anchor`/`prompt`/`completion`:

| target | 8-gram hits | |
|---|---:|---|
| teleqna | 530 / 10,000 | 5.30% |
| srsRANBench | 2 / 1,502 | 0.13% |
| TeleLogs official | **0 / 864** | 0.00% |

Every inspected example was a shared multiple-choice **stem**, not a shared question:

| matched shingle | test question | corpus record |
|---|---|---|
| `which of the following is not a benefit` | …of applying deep learning to wireless networking | …of using declarative configurations in Kubernetes |
| `what is the term used to refer to` | …management and orchestration of services | …combined PDB and interference estimation in GSC |
| `what is the effect of increasing the number` | …of antennas at the macro BS | …of guard subcarriers between 4-PRB subbands |

k=8 is wide enough to catch the formula and narrow enough to stop before the
specifics. So the count is an **upper** bound on leakage in exactly the way it is a
lower bound on coverage: it over-reports when targets share phrasing.

`verify_contamination.py` indexes the last 8 words instead — a question's tail
carries its specifics, not its formula — then confirms by requiring the whole
normalised question verbatim in the record:

| tier | count | of 10,000 |
|---|---:|---:|
| tail hit | 28 | 0.28% |
| **whole question verbatim** | **15** | **0.15%** |
| question + gold option text in same record | 8 | 0.08% |

The 15 are questions any generator would produce from the same paragraph — *"What
is the definition of the Internet of Things (IoT)?"*, *"What is the key challenge
for 5G and beyond-5G wireless networks?"*. TeleQnA was generated by an LLM reading
IEEE papers; OTel's set was generated by an LLM reading the same IEEE papers.
Convergent generation, not copying. Drop all 28 tail hits from any training use
anyway — it costs nothing.

### What the OTel set actually trains

| | |
|---|---|
| records | 606,237 — 265.1M prompt tokens, 19.9M completion |
| shape | `anchor` question → RAG prompt with `CONTEXT 1..N` → grounded answer |
| prompt_type | `prompt_0` 300,000 / `prompt_1` 300,000 / `direct_qa` 6,237 |
| positive chunks | 1 for 500,000 records; **0 for 106,237** |
| negative chunks | 4 for 357,828, ranging to 9 |
| abstention | **100,008 records (16.5%)** — answer is "not in the context" |
| completion length | median 29 tokens |
| **MCQ-shaped answers** | **154 (0.03%)** |

Two things follow. First, they train retrieval-miss robustness directly: every
record carries 4–9 deliberately wrong chunks, and one in six has no correct chunk
at all and must abstain. That is the failure mode the oracle-ladder arm M was
designed to *measure*; here it is training data. Second, this set is essentially
free of multiple choice — yet the model built on it scores 0.912 on an MCQ
benchmark. Grounded open-ended QA transfers to the MCQ format without MCQ training,
so a train set for this track does not need to be MCQ either.

Contrast with AdaptKey: numbered option list, TeleQnA's own category label, gold
option as the SFT target, ~9,986/10,000. Two published telecom corpora, opposite
verdicts, same scan.

## Files

| file | what it does |
|---|---|
| `profile_data.py` | reads the parquet the way the harness does, writes `data/teleqna/test.jsonl`, prints the profile above |
| `run_baseline.py` | Inspect-faithful scorer + Qwen3-8B runner; control arms `--always`, `--heuristic`, `--permute`; per-subject and per-choice-count breakdowns |
| `make_probe_set.py` | builds the paired `orig` / `perm` / `distract` / `paraphrase` arms; validates every paraphrase and records what it drops and why |
| `run_tsguess.py` | masks one option and scores reconstruction; `--mask distractor` is the evidence arm, `--mask gold` the comparison |
| `make_selfgen_control.py` | the valid guessability control (see above) |
| `compare_probe.py` | paired McNemar across arms, plus the TS-Guessing summaries |
| `make_splits.py` | 400 / 600 / 9,000 stratified by subject, deterministic, asserts disjointness and no lost rows |
| `dspy_optimize_prompt.py` | GEPA/COPRO over the system prompt; `ServingAdapter` keeps the harness message byte-identical so the artefact deploys as measured |
| `scan_contamination.py` | 8-gram shingle scan of any corpus (jsonl / parquet / directory of files) against any test set; run it before using anything third-party |
| `verify_contamination.py` | second stage — indexes question tails and requires the whole question verbatim, separating real leakage from shared multiple-choice stems |
| `attribute_gain.py` | splits any delta four ways — by subject, shape-cued vs not, agreement with the shape heuristic, and knowledge wins — and refuses to call a gain real if it lands only where the artefact already wins |
| `jobs/*.sh` | H200 client-pod wrappers (drop into `client_jobs*`, read `client_done*`) |

```bash
.venv-srsran/bin/python infra/teleqna/profile_data.py --samples 4
.venv-srsran/bin/python infra/teleqna/run_baseline.py \
    --data data/teleqna/test.jsonl --out /tmp/heur --heuristic
```

Self-check, run locally with no model, reproducing the numbers above:

| arm | score |
|---|---:|
| `--always A` | 0.2210 |
| `--always B` | 0.2179 |
| `--heuristic` | 0.3972 |
| `--heuristic --permute` | 0.3942 |

## Open decisions

Three of the original four are now closed by measurement and written up above:
the baseline is run (thinking helps, +2.02 pp, unlike srsranbench where it
hurt), the memorisation gate is passed, and prompt optimisation is dead —
including its leak-everything ceiling.

What remains open:

1. **Where the remaining points are.** 2,805 wrong rows at 71.95%. Standards
   specifications alone is 772 of them (61.40% on 2,000 rows) and is the one
   subject proven unreachable by both prompting and deliberation. Research
   publications is 1,132 wrong on 4,500 but already the second-best subject.
2. **Retrieval, now that it is confirmed permitted.** We serve the model, so
   the request can be rewritten and augmented before it reaches the weights;
   the harness cannot tell and the leaderboard row is self-reported. This is
   the only intervention aimed at the actual bottleneck. Requires a corpus:
   3GPP Rel-17/18 + IEEE for the ~3,000 standards rows, telecom literature for
   the ~6,500 research rows.
3. **Self-consistency voting.** Cheap, structural, and independent of
   knowledge — but the rotation result predicts what it buys. 11% of rows sit
   on a knife edge and vote roughly 50/50, so expect ≈ +1 pp and treat anything
   larger as suspicious. Worth running as a control, not as a plan.
4. **RAG as teacher rather than as scaffold.** If retrieval works, distilling
   its answers into the weights removes the serving-time dependency. Only
   sequenced after (2) shows there is anything to distil. Note the memorisation
   probe must be re-run against any fine-tuned checkpoint — training is where
   contamination would newly enter.
3. **Where the corpus comes from.** Everything the answers are drawn from is
   public and legitimate to train on — 3GPP specifications (the `[Release NN]`
   tags say which), IEEE standards, and telecom literature. That is exactly what
   AT&T, Tele-LLMs and TelecomGPT did. What is *not* legitimate is the 10,000
   rows in `data/teleqna/`.
4. **Retrieval: as teacher, or as scaffold?** An earlier draft of this file
   asserted that the leaderboard "takes models, not pipelines". **That is
   wrong**, and it is corrected here rather than deleted, because it was about
   to steer the whole track.

   Checked against source. `gsma-labs/satellite` — GSMA's own submission TUI —
   ships an `open-local` provider category whose entries (`vllm`, `sglang`,
   `llama-cpp`, `ollama`) are configured by nothing but a base URL
   (`VLLM_BASE_URL`, default `http://localhost:8000/v1`). Inspect calls it over
   OpenAI chat-completions. Anything that speaks that protocol is a "model" as
   far as the harness is concerned; nothing inspects what is behind the URL.
   Submission is `parquet_builder.py` reading `(accuracy, stderr, n_samples)`
   out of the local Inspect log headers into a one-row parquet, then a pull
   request to `gsma-labs/leaderboard`. No trajectories, no attestation, no
   weight verification. Scores are self-reported and human-reviewed.

   So both routes are open, and the choice is about what the number should
   mean, not about what is permitted:

   - **Scaffold** — serve retrieval + verification behind the endpoint. Highest
     achievable score; matches how the system would actually be deployed;
     Telco-oRAG measured +10.6 points on Lexicon from context alone. Costs:
     the leaderboard row names an artefact that does not reproduce on its own
     (someone pulling `Qwen3-8B` gets 71.95%, not the submitted number), and
     the per-case call count is real — TeleLogs' shipped pipeline runs 11.61 LM
     calls per case, which on 10,000 rows is ~116k calls plus retrieval.
   - **Teacher** — RAG generates the training data, the knowledge is distilled
     into weights, the submitted endpoint is one forward pass. Lower ceiling per
     unit of effort, but the row names something that reproduces, and the gains
     transfer to the other six benchmarks that the leaderboard `average` ranks
     on. This is the ORANSight-2.0 / Tele-LLMs / AT&T recipe.

   Note also that AT&T's own OTel repo describes its stack as "retrieval,
   reranking, context-grounded generation, and abstention models" and states
   that "LLM results evaluate context-grounded RAG behavior, not unrestricted
   context-free QA" — said about their internal evaluation partitions rather
   than about their leaderboard row, so it is a strong hint and not proof that
   the #1 entry is a system rather than a bare model.

   These are not exclusive: the teacher route produces weights that also work
   inside a scaffold, so doing it in that order keeps both options.
5. **Subject-weighted effort.** Research publications (4,500) + Standards
   specifications (2,000) are 65% of the benchmark and the two hardest subjects.
   Lexicon (500) is 5% and already near-solved. Effort should follow the rows.
