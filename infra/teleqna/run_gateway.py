#!/usr/bin/env python3
"""Closed-book teleqna eval against the Viettel gateway (OpenAI-compatible).

Same prompt, parser and scoring as run_baseline.py (= the GSMA Inspect
harness); only the transport differs:

  * bearer auth from $GATEWAY_KEY (never written to any file or log)
  * thinking is toggled with the TOP-LEVEL `enable_thinking` field — the
    gateway drops `chat_template_kwargs`, measured 2026-08-27 (5 completion
    tokens either way through that route; 700-1000 through the top-level flag)
  * reasoning comes back in `message.reasoning` (not `reasoning_content`)

Resumable on results.jsonl. Summary written to summary.json.
"""
from __future__ import annotations

import argparse, concurrent.futures, json, os, sys, threading, time, urllib.request, urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baseline import build_sample, build_prompt, parse_answer, summarize, now_iso  # noqa: E402

URL = os.environ.get("GATEWAY_URL", "https://stream-netmind.viettel.vn/aigw/ai/v1/chat/completions")
KEY = os.environ.get("GATEWAY_KEY", "")


def request_one(sample, args):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": build_prompt(sample, False)}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": 42,
    }
    # The gateway treats the mere PRESENCE of `enable_thinking` as "on":
    # sending `false` still produced reasoning (247 chars, truncated at 64
    # tokens on all 100 pilot rows). Omit the field for no-think.
    if args.thinking:
        payload["enable_thinking"] = True
    started = time.monotonic()
    last = ""
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(URL, data=json.dumps(payload).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                raw = json.load(r)
            msg = raw["choices"][0]["message"]
            completion = msg.get("content") or ""
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            parsed = parse_answer(completion, len(sample["choices"]))
            return {
                "sample_id": sample["sample_id"], "sample_index": sample["sample_index"],
                "subject": sample["subject"], "target": sample["target"],
                "gold_index": sample["gold_index"], "original_gold_index": sample["original_gold_index"],
                "n_choices": len(sample["choices"]), "parsed_answer": parsed,
                "correct": bool(parsed) and parsed == sample["target"], "parse_failed": not parsed,
                "finish_reason": raw["choices"][0].get("finish_reason"),
                "completion": completion, "reasoning_chars": len(reasoning),
                "reasoning": reasoning if args.keep_reasoning else "",
                "usage": raw.get("usage", {}), "elapsed_seconds": round(time.monotonic() - started, 3),
                "completed_at": now_iso(), "error": None,
            }
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200]!r}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)[:200]
        time.sleep(min(60, 3 * attempt))
    return {"sample_id": sample["sample_id"], "sample_index": sample["sample_index"], "subject": sample["subject"],
            "target": sample["target"], "gold_index": sample["gold_index"], "original_gold_index": sample["original_gold_index"],
            "n_choices": len(sample["choices"]), "parsed_answer": "", "correct": False, "parse_failed": True,
            "finish_reason": None, "completion": "", "reasoning_chars": 0, "reasoning": "", "usage": {},
            "elapsed_seconds": round(time.monotonic() - started, 3), "completed_at": now_iso(), "error": last}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B-FP8")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--keep-reasoning", action="store_true")
    args = p.parse_args()
    if not KEY:
        sys.exit("GATEWAY_KEY not set")
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    samples = [build_sample(r, False) for r in rows]
    args.out.mkdir(parents=True, exist_ok=True)
    res_path = args.out / "results.jsonl"
    done = {}
    if res_path.exists():
        for l in open(res_path):
            if l.strip():
                r = json.loads(l)
                if r.get("error") is None:
                    done[r["sample_id"]] = r
    todo = [s for s in samples if s["sample_id"] not in done]
    print(f"{len(done)} done, {len(todo)} to run, thinking={args.thinking} model={args.model}", flush=True)
    lock = threading.Lock()
    t0 = time.monotonic()
    with open(res_path, "a") as fh, concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        for n, rec in enumerate(ex.map(lambda s: request_one(s, args), todo), 1):
            with lock:
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                done[rec["sample_id"]] = rec
            if n % 100 == 0 or n == len(todo):
                c = sum(r["correct"] for r in done.values())
                print(f"[{n}/{len(todo)}] acc so far {100*c/len(done):.2f}% ({len(done)} rows) "
                      f"{(time.monotonic()-t0)/n:.2f}s/row errs={sum(1 for r in done.values() if r['error'])}", flush=True)
    records = [done[s["sample_id"]] for s in samples if s["sample_id"] in done]
    summary = summarize(records)
    summary.update(model=args.model, thinking=args.thinking, max_tokens=args.max_tokens, n=len(records))
    json.dump(summary, open(args.out / "summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
