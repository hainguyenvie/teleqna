#!/usr/bin/env python3
"""Compose two LoRA adapters in weight space instead of in the training data.

Arm J mixed the two training sets and got arm G-full back: it recovered 17 of
the 203 rows only arm H reached, and 2 of the 52 on C_hard. Same epochs, so not
an exposure problem -- when the real labels hold 70% of the loss they win the
capacity competition and the synthetic facts are never consolidated.

Weight-space composition has no gradients to compete. LoRA is dW = (alpha/r)*B@A,
so stacking the ranks is EXACT, not an approximation:

    A_cat = [A_G ; A_H]        (r_G + r_H, in)
    B_cat = [B_G , s*B_H]      (out, r_G + r_H)
    dW    = (alpha_new/r_new) * B_cat @ A_cat
          = (alpha_G/r_G)*B_G@A_G  +  s*(alpha_H/r_H)*B_H@A_H

with alpha_new/r_new chosen equal to alpha_G/r_G (256/128 == 128/64 == 2), so
the arm G-full half is reproduced bit for bit at s=0 and s just dials arm H in.

This is the last cheap question on the branch: do the two deltas coexist, or
does arm H's drift come back the moment its facts do?
"""
import argparse, json, os, shutil
import torch
from safetensors.torch import load_file, save_file

M = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft/models")
ap = argparse.ArgumentParser()
ap.add_argument("--a", default=f"{M}/ctxdistill-armG_full-adapter")
ap.add_argument("--b", default=f"{M}/ctxdistill-armH-adapter")
ap.add_argument("--scale", type=float, required=True, help="weight on adapter b")
ap.add_argument("--out", required=True)
a = ap.parse_args()

ca = json.load(open(f"{a.a}/adapter_config.json"))
cb = json.load(open(f"{a.b}/adapter_config.json"))
for k in ["target_modules", "peft_type", "task_type", "use_rslora"]:
    va, vb = ca.get(k), cb.get(k)
    if isinstance(va, list): va, vb = sorted(va), sorted(vb)
    assert va == vb, f"adapters disagree on {k}: {va} vs {vb}"
assert not ca.get("use_rslora"), "rslora changes the scaling; this maths assumes alpha/r"
ra, rb = ca["r"], cb["r"]
sa, sb = ca["lora_alpha"] / ra, cb["lora_alpha"] / rb

r_new = ra + rb
alpha_new = sa * r_new          # keep adapter a's effective scaling exactly
s_new = alpha_new / r_new
assert abs(s_new - sa) < 1e-9

wa = load_file(f"{a.a}/adapter_model.safetensors")
wb = load_file(f"{a.b}/adapter_model.safetensors")
assert set(wa) == set(wb), f"key mismatch: {len(set(wa) ^ set(wb))} keys differ"

out, n_a, n_b = {}, 0, 0
for k, ta in wa.items():
    tb = wb[k]
    if ".lora_A" in k:
        assert ta.shape[1] == tb.shape[1], f"{k}: input dims differ"
        out[k] = torch.cat([ta.float(), tb.float()], dim=0).to(ta.dtype)
        n_a += 1
    elif ".lora_B" in k:
        assert ta.shape[0] == tb.shape[0], f"{k}: output dims differ"
        # a's half is rescaled by sa/s_new == 1; b's carries the user's scale and
        # its own alpha/r, converted into the merged adapter's scaling.
        out[k] = torch.cat([ta.float() * (sa / s_new),
                            tb.float() * (a.scale * sb / s_new)], dim=1).to(ta.dtype)
        n_b += 1
    else:
        raise SystemExit(f"unexpected tensor in a LoRA checkpoint: {k}")

os.makedirs(a.out, exist_ok=True)
save_file(out, f"{a.out}/adapter_model.safetensors")
cfg = dict(ca); cfg["r"] = r_new; cfg["lora_alpha"] = alpha_new
json.dump(cfg, open(f"{a.out}/adapter_config.json", "w"), indent=2)
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
    if os.path.exists(f"{a.a}/{f}"):
        shutil.copy(f"{a.a}/{f}", f"{a.out}/{f}")

print(f"{a.out}")
print(f"  r {ra}+{rb} -> {r_new}, alpha {ca['lora_alpha']} -> {alpha_new} "
      f"(scaling {sa} held constant)")
print(f"  {n_a} A tensors, {n_b} B tensors; arm H weighted {a.scale}")

# a numerical check on one module: does the merged dW equal the sum it claims?
k = next(k for k in wa if k.endswith("lora_A.weight"))
kb_ = k.replace("lora_A", "lora_B")
want = sa * (wa[kb_].float() @ wa[k].float()) + \
       a.scale * sb * (wb[kb_].float() @ wb[k].float())
got = s_new * (out[kb_].float() @ out[k].float())
err = (want - got).abs().max().item()
print(f"  check on {k.split('base_model.model.')[-1][:44]}: max|dW_want - dW_got| = {err:.3e}")
assert err < 1e-3, "merged delta does not reproduce the sum"
