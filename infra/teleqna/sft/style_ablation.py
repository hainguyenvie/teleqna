#!/usr/bin/env python3
"""Does the trained answer *style* explain the -7 points, or did the weights lose knowledge?

armCE lost 70 net rows against base. The anatomy says the loss is concentrated
where the eval offers an aggregate option: "all of the above" rows fell 96.0 ->
56.0 on n=50, "none of the above" 86.6 -> 68.7 on n=67. Together that is half
the damage from 12% of the eval. The training mix contains one single
all-of-the-above item out of 52,568 MCQs, and 96.8% of its MCQ completions
follow one rigid schema: assert the gold fact in the first sentence, then refute
every other option by letter. That schema cannot ever produce "all of the
above" -- refuting the others is a step in it.

So the hypothesis is that we did not overwrite what the model knows, we
overwrote how it answers, and the how forbids a class of answers.

The test needs no training. Force the schema onto the *untrained* base with a
one-shot demonstration and see whether it falls the same way, then hand the
trained adapter a demonstration in the base's own style and see whether it comes
back. Both demos carry the identical question, options and gold fact; they
differ only in whether the answer is delivered as "assert then refute each
letter" or as the base's own deliberation. Arm B exists so that arm C is read
against a one-shot control rather than against zero-shot -- otherwise "having an
example at all" would be confounded with the style of that example.

  A  base    standard prompt          control, must land on 80.10
  B  base    one-shot, base style     controls for the presence of an example
  C  base    one-shot, refute style   C ~ B means style is harmless
                                      C ~ 73 means style is the whole disease
  D  armCE   standard prompt          control, must land on 73.10
  E  armCE   one-shot, base style     recovery says the knowledge is intact

The final user turn stays byte-identical to the harness template in every arm;
the demonstration is a prior turn pair, so nothing about the question the model
is actually scored on changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dev_vllm import build_prompt, parse, parse_official  # noqa: E402

# A real training row, verbatim, so the forced style is the trained style and
# not my impression of it.
DEMO_Q = {
    "question": "What message does the SN use to acknowledge an MN initiated "
                "SN Modification?",
    "choices": ["SgNB Addition Request",
                "SgNB Modification Request Acknowledge",
                "SgNB Modification Required",
                "SgNB Reconfiguration Complete"],
}
DEMO_REFUTE = (
    "The SN responds with the SgNB Modification Request Acknowledge message, "
    "which may contain SCG radio resource configuration information within a "
    "NR RRC configuration message. A) SgNB Addition Request is wrong: This is "
    "a message for adding a secondary node, not modifying it. C) SgNB "
    "Modification Required is wrong: This is the message sent by the SN to "
    "request modification, not acknowledge. D) SgNB Reconfiguration Complete "
    "is wrong: This is a message confirming successful reconfiguration, not "
    "acknowledgment.\n\nANSWER: B"
)
# The same facts, delivered the way the base delivers them: name the answer,
# explain it, and only then note what the others are -- no per-letter verdict
# schema, and no commitment that the answer must be exactly one of the listed
# facts.
DEMO_BASE = (
    "The correct answer is B) SgNB Modification Request Acknowledge. When the "
    "MN initiates an SN Modification, the SN acknowledges it with the SgNB "
    "Modification Request Acknowledge message, which may carry SCG radio "
    "resource configuration information inside an NR RRC configuration "
    "message. The other options describe different steps of the procedure: "
    "addition rather than modification, the SN-initiated request, and the "
    "confirmation that reconfiguration finished.\n\nANSWER: B"
)

ARMS = [("A_base_plain", None, None),
        ("B_base_demo_base", None, "base"),
        ("C_base_demo_refute", None, "refute"),
        ("D_ce_plain", "armCE", None),
        ("E_ce_demo_base", "armCE", "base")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True, help="path to the armCE adapter")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = [json.loads(l) for l in args.data.open(encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    demo_user = build_prompt(DEMO_Q)

    def prompts_for(style):
        head = []
        if style:
            head = [{"role": "user", "content": demo_user},
                    {"role": "assistant",
                     "content": DEMO_REFUTE if style == "refute" else DEMO_BASE}]
        return [tok.apply_chat_template(
            head + [{"role": "user", "content": build_prompt(r)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for r in rows]

    llm = LLM(model=args.base, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, tensor_parallel_size=1,
              enable_lora=True, max_lora_rank=64, enforce_eager=False)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new)

    def kind(r):
        t = r["choices"][-1].strip().lower().rstrip(".")
        return ("aota" if t.startswith("all of") else
                "nota" if t.startswith("none of") else "plain")

    out = {}
    for name, adapter, style in ARMS:
        lreq = LoRARequest("ce", 1, args.adapter) if adapter else None
        outs = llm.generate(prompts_for(style), sp, lora_request=lreq)
        res, by_kind = [], {}
        for r, o in zip(rows, outs):
            comp = o.outputs[0].text
            n = len(r["choices"])
            gold = chr(65 + int(r["answer"]))
            got = parse_official(comp, n)
            ok = got == gold
            by_kind.setdefault(kind(r), []).append(int(ok))
            res.append({"sample_id": r["sample_id"], "parsed": got,
                        "parsed_lenient": parse(comp, n), "correct": bool(ok),
                        "completion": comp})
        acc = sum(x["correct"] for x in res) / len(res)
        out[name] = {"arm": name, "adapter": adapter, "style": style,
                     "accuracy": round(acc, 4),
                     "correct": sum(x["correct"] for x in res),
                     "unparsed": sum(1 for x in res if not x["parsed"]),
                     "by_kind": {k: {"n": len(v), "acc": round(sum(v)/len(v), 4)}
                                 for k, v in sorted(by_kind.items())},
                     "results": res}
        print(f"  {name:20s} acc={acc:.4f} unparsed={out[name]['unparsed']} "
              f"{ {k: v['acc'] for k, v in out[name]['by_kind'].items()} }",
              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
