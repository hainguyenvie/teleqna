#!/usr/bin/env python3
"""Compact decision-ladder figure.

Text is wrapped against real metrics (the .ax class is 11px monospace =>
6.6px/char; .lbl em is 12.5px bold sans => ~6.8px/char), box heights follow
the wrapped line count, and row heights follow the tallest box — so nothing
can overflow a border or collide with a neighbour. No leader lines cross the
question labels: the spine carries the eye instead.
"""

CH_MONO, CH_BOLD, CH_LBL = 6.6, 6.85, 6.3
W, GAP = 900.0, 12.0
LABEL_X, BOXES_X0, BOXES_X1 = 34.0, 222.0, W - 12
AREA = BOXES_X1 - BOXES_X0
PAD = 13.0
TITLE_BASE, FIRST_LINE, LINE_H, BOTTOM_PAD = 21.0, 38.0, 14.0, 13.0
ROW_GAP, TOP = 20.0, 34.0


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wrap(text, width_px, char_px):
    limit = max(8, int(width_px / char_px))
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if len(trial) <= limit:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


ROWS = [
    (["Q1 · is the 71.95%", "just memorised?"], [
        ("no", "NO — four independent probes agree", [
            "paraphrase −0.87pp (p=0.55) · rotation is an order effect, not recall",
            "TS-guessing recall falls as the masked choice gets longer = guessing",
        ]),
    ]),
    (["Q2 · knowledge gap or", "discrimination gap?"], [
        ("yes", "DISCRIMINATION — for most of the errors", [
            "distract probe +15.83pp (p<0.001): the gold answer is already known",
            "exception: Standards-spec gains only +0.35 from thinking = a real gap",
        ]),
    ]),
    (["Q3 · can the cheap", "eval-time layer", "harvest it?"], [
        ("no", "Prompt search · DSPy → GEPA", [
            "held-out −1.12pp; even the answer-key leak ceiling is −0.45",
            "!CLOSED",
        ]),
        ("no", "Self-consistency vote ×4", [
            "+0.74pp for 4× the cost; the 83.85 oracle stays out of reach",
            "!CLOSED",
        ]),
        ("yes", "Thinking mode", [
            "+2.02pp and order-robust, but only +0.35 where it matters",
            "!KEPT AS SERVING MODE",
        ]),
    ]),
    (["Q4 · which weight", "channel teaches", "discrimination?"], [
        ("no", "SFT on gold letters", [
            "−3.6pp at the gentlest dose; breaks 3 rows per 2 taught",
            "!FAILED — a method problem, not a dose problem",
        ]),
        ("yes", "Single-token DPO + anchors", [
            "KL anchor caps the damage; best 76.50 thinking (+1.4)",
            "!KEPT — but the net stalls at +4..6 rows",
        ]),
    ]),
    (["Q5 · why does DPO", "plateau there?"], [
        ("amber", "The letter channel carries ~1 bit per question", [
            "fix-rate is FLAT against training-pair similarity (23.7 / 22.7 / 24.3 / 21.4%)",
            "164 dev rows stay wrong in all four arms — calibration moved, knowledge did not",
        ]),
    ]),
    (["Q6 · which channel", "actually carries", "facts?"], [
        ("amber", "CPT on source text", [
            "192.6M tokens, error-targeted; the AT&T lever at 1/2000 dose",
            "!RUNNING NOW",
        ]),
        ("plan", "GRPO on thinking traces", [
            "rewards whole reasoning paths; closes pass@k → pass@1",
            "!PLANNED · TRL staged",
        ]),
        ("no", "RAG at inference", [
            "the strongest scorer on paper; closed-book track forbids it",
            "!BLOCKED BY RULES",
        ]),
    ]),
]

STYLE = {
    "no": ("color-mix(in srgb,var(--crit) 8%,transparent)", "var(--crit)"),
    "yes": ("color-mix(in srgb,var(--good) 8%,transparent)", "var(--good)"),
    "amber": ("color-mix(in srgb,var(--warn) 10%,transparent)", "var(--warn)"),
    "plan": ("none", "var(--s1)"),
}

# ── layout pass: wrap everything, size every box, then size every row ──────
laid = []
for qlines, boxes in ROWS:
    n = len(boxes)
    bw = (AREA - GAP * (n - 1)) / n
    inner = bw - 2 * PAD
    prepared, tallest = [], 0.0
    for kind, title, details in boxes:
        tl = wrap(title, inner, CH_BOLD)
        body = []
        for d in details:
            if d.startswith("!"):
                body += [("v", ln) for ln in wrap(d[1:], inner, CH_MONO)]
            else:
                body += [("d", ln) for ln in wrap(d, inner, CH_MONO)]
        n_title_extra = len(tl) - 1
        h = (FIRST_LINE + n_title_extra * 16 + LINE_H * (len(body) - 1)
             + BOTTOM_PAD)
        prepared.append((kind, tl, body, h))
        tallest = max(tallest, h)
    laid.append((qlines, prepared, bw, tallest))

parts, y, centers = [], TOP, []
for qlines, prepared, bw, row_h in laid:
    cy = y + row_h / 2
    centers.append(cy)
    ty = cy - (len(qlines) - 1) * 8 + 4
    for i, ln in enumerate(qlines):
        cls = "lbl em" if i == 0 else "lbl"
        parts.append(f'<text class="{cls}" x="{LABEL_X}" y="{round(ty,1)}">{esc(ln)}</text>')
        ty += 16
    parts.append(f'<circle cx="18" cy="{round(cy,1)}" r="4.5" fill="var(--panel)" '
                 f'stroke="var(--s1)" stroke-width="2"/>')
    for i, (kind, tl, body, h) in enumerate(prepared):
        bx = BOXES_X0 + i * (bw + GAP)
        fill, stroke = STYLE[kind]
        dash = ' stroke-dasharray="5 4"' if kind == "plan" else ""
        parts.append(f'<rect x="{round(bx,1)}" y="{round(y,1)}" width="{round(bw,1)}" '
                     f'height="{round(row_h,1)}" rx="10" fill="{fill}" stroke="{stroke}"{dash}/>')
        tb = y + TITLE_BASE
        for ln in tl:
            parts.append(f'<text class="lbl em" x="{round(bx+PAD,1)}" y="{round(tb,1)}">{esc(ln)}</text>')
            tb += 16
        dy = y + FIRST_LINE + (len(tl) - 1) * 16
        for kindln, ln in body:
            if kindln == "v":
                parts.append(f'<text class="floor-lbl" x="{round(bx+PAD,1)}" y="{round(dy,1)}" '
                             f'style="fill:{stroke}">{esc(ln)}</text>')
            else:
                parts.append(f'<text class="ax" x="{round(bx+PAD,1)}" y="{round(dy,1)}">{esc(ln)}</text>')
            dy += LINE_H
    y += row_h + ROW_GAP

parts.insert(0, f'<line x1="18" y1="{round(centers[0],1)}" x2="18" y2="{round(centers[-1],1)}" '
                f'stroke="var(--rule)" stroke-width="1.6"/>')
H = round(y - ROW_GAP + 14, 1)
svg = (f'<svg viewBox="0 0 {int(W)} {H}" role="img" '
       f'aria-label="Decision ladder: six questions, each answered by a measurement">\n'
       + "\n".join(parts) + "\n</svg>")
open("fig_tree.svg", "w").write(svg)
print("tree height", H)
for (q, prep, bw, rh) in laid:
    print(f"  row {q[0][:14]:16s} bw={bw:6.1f} inner={bw-26:6.1f} h={rh:5.1f} "
          f"lines={[len(b[2]) for b in prep]}")
