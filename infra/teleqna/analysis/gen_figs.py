#!/usr/bin/env python3
"""Generate the two big SVG figures for journey.html: the all-methods
scoreboard and the decision tree. Coordinates computed, not hand-guessed."""

X0, X1 = 366.0, 1140.0
V0, V1 = 20.0, 95.0
SCALE = (X1 - X0) / (V1 - V0)
W = 1300.0


def esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def x(v):
    return round(X0 + (v - V0) * SCALE, 1)


BANDS = [
    ("References & controls — what the number means", [
        ("always answer A", 22.10, "floor", "no model · 10k"),
        ("distractor-shape heuristic", 39.72, "floor", "no model · 10k"),
        ("human telecom expert (paper)", 64.86, "floor", "published"),
        ("GPT-4, TeleQnA paper 2023", 74.91, "floor", "published"),
        ("AT&T OTel-LLM-8.3B-QnA — SOTA", 91.20, "sota", "400B-token CPT · published"),
    ]),
    ("Eval-time layer — full 10,000 / held-out 9,000, no weights touched", [
        ("baseline no-think", 71.95, "base", "10k · reference point"),
        ("baseline no-think, rotated choices", 70.36, "neg", "−1.59 · 1,135 rows flip"),
        ("+ seed system prompt (hand-written)", 71.40, "neg", "9k held-out · −0.34, p=0.29"),
        ("+ DSPy/GEPA optimised prompt", 70.62, "neg", "9k held-out · −1.12, p=0.001"),
        ("+ GEPA fit on 10k, scored on 10k (leak ceiling)", 71.50, "neg",
         "−0.45 even with the answer key"),
        ("+ thinking mode", 73.97, "pos", "10k · +2.02, p<0.001 — kept as serving mode"),
        ("+ thinking, rotated choices", 73.58, "flat", "−0.39, p=0.24 — order-robust"),
        ("+ self-consistency vote, 4 permutations", 74.71, "flat", "+0.74 over thinking · 4× cost"),
        ("oracle: correct in ANY branch", 83.85, "oracle", "ceiling of all ensembling — not submittable"),
    ]),
    ("Other models, measured on the same harness", [
        ("OTel-LLM-8B-IT (AT&T's released model)", 60.87, "neg", "10k · −11.1 vs base"),
        ("Qwen3-235B-A22B", 79.70, "flat", "+5.2pp for 30× params"),
    ]),
    ("Weights layer — dev-1000, single A/A-calibrated stack", [
        ("SFT, gentlest dose (0.24 epoch)", 70.20, "neg", "−3.6 · damage at every dose"),
        ("base · no-think", 74.00, "base", "A/A reference"),
        ("DPO3@200 · no-think", 74.40, "pos", "+0.4"),
        ("base · thinking", 75.10, "base", "A/A reference"),
        ("DPO3@200 · thinking  ← best so far", 76.50, "best", "+1.4 · 52 fixed / 38 broken"),
        ("CPT campaign 1 · ~200M tokens", None, "running", "training now — ETA 11:45 UTC"),
    ]),
]

FILL = {
    "floor": "var(--floor)", "sota": "var(--warn)", "base": "var(--ink-3)",
    "neg": "var(--crit)", "pos": "var(--good)", "flat": "var(--s2)",
    "oracle": "var(--warn)", "best": "var(--s1)", "running": "var(--s1)",
}

ROW, BAND_H, TOP = 23.0, 34.0, 34.0
out = []
y = TOP
for title, rows in BANDS:
    out.append(f'<text class="lbl em" x="20" y="{round(y,1)}">{esc(title)}</text>')
    y += BAND_H - 12
    for label, val, kind, note in rows:
        by = round(y, 1)
        out.append(f'<text class="ax" x="{X0-14}" y="{round(by+11,1)}" text-anchor="end">{esc(label)}</text>')
        if val is None:
            out.append(f'<rect x="{X0}" y="{by}" width="120" height="15" rx="3" '
                       f'fill="none" stroke="{FILL[kind]}" stroke-dasharray="4 3"/>')
            out.append(f'<text class="ax" x="{X0+130}" y="{round(by+12,1)}" '
                       f'style="fill:var(--s1)">running · {esc(note)}</text>')
        else:
            w = round(x(val) - X0, 1)
            out.append(f'<rect x="{X0}" y="{by}" width="{w}" height="15" rx="3" fill="{FILL[kind]}"/>')
            tail = W - x(val) - 12
            if tail < 52 + len(note) * 6.6:
                out.append(f'<text class="val" x="{round(x(val)-8,1)}" y="{round(by+12,1)}" '
                           f'text-anchor="end" style="fill:var(--plane)">{val:.2f}</text>')
                out.append(f'<text class="ax" x="{round(x(val)-50,1)}" y="{round(by+12,1)}" '
                           f'text-anchor="end" style="fill:var(--plane);opacity:.82">{esc(note)}</text>')
            else:
                out.append(f'<text class="val" x="{round(x(val)+7,1)}" y="{round(by+12,1)}">{val:.2f}</text>')
                out.append(f'<text class="ax" x="{round(x(val)+50,1)}" y="{round(by+12,1)}">{esc(note)}</text>')
        y += ROW
    y += 10

H = round(y + 26, 1)
grid = []
for v in range(20, 100, 10):
    grid.append(f'<line class="grid-line" x1="{x(v)}" y1="24" x2="{x(v)}" y2="{round(y-8,1)}"/>')
    grid.append(f'<text class="ax" x="{x(v)}" y="{H-6}" text-anchor="middle">{v}</text>')

svg = (f'<svg viewBox="0 0 {int(W)} {H}" role="img" aria-label="Every method measured on this track">\n'
       + "\n".join(grid) + "\n" + "\n".join(out) + "\n</svg>")
open("fig_scoreboard.svg", "w").write(svg)
print("scoreboard height", H)

# (the decision ladder now lives in gen_tree.py — it superseded the old tree)
