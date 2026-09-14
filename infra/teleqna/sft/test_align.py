#!/usr/bin/env python3
"""Check the right-alignment arithmetic without loading a model.

The whole trainer rests on one claim: that after left padding, slicing the last
width+1 columns off two sequences of different length makes their final tokens
line up, and that the completion mask then selects exactly the answer positions.
If that claim is wrong the KL term compares a student's answer against a
teacher's prompt and the loss still goes down, so it has to be checked directly
rather than inferred from a training curve.
"""
import torch

MAXCOMP = 8


def tail(x, width):
    return x[:, -(width + 1):-1, :]


def comp_mask(n_comp, width, device):
    idx = torch.arange(width, device=device).unsqueeze(0)
    return idx >= (width - n_comp.to(device).unsqueeze(1))


def pad_left(seqs, pad=0):
    m = max(len(s) for s in seqs)
    ids = torch.tensor([[pad] * (m - len(s)) + s for s in seqs])
    att = torch.tensor([[0] * (m - len(s)) + [1] * len(s) for s in seqs])
    pos = (att.cumsum(-1) - 1).clamp_min(0)
    return ids, att, pos


# Two rows. Answer tokens are the 900-block; prompt tokens are 100s; the teacher
# carries an extra context prefix of 700s, of different length in each row.
student = [[101, 102, 103, 901, 902, 903],
           [101, 102, 901, 902]]
teacher = [[701, 702, 703, 704, 101, 102, 103, 901, 902, 903],
           [701, 702, 101, 102, 901, 902]]
n_comp = torch.tensor([3, 2])

s_ids, s_att, s_pos = pad_left(student)
t_ids, t_att, t_pos = pad_left(teacher)
print("student ids\n", s_ids)
print("teacher ids\n", t_ids)
print("student position_ids\n", s_pos)

width = min(MAXCOMP, s_ids.size(1) - 1)
V = 1000
# Stand-in for logits: one-hot on the token that position actually holds, so a
# misalignment shows up as comparing different token ids.
s_log = torch.nn.functional.one_hot(s_ids, V).float()
t_log = torch.nn.functional.one_hot(t_ids, V).float()

s_tok = tail(s_log, width).argmax(-1)
t_tok = tail(t_log, width).argmax(-1)
mask = comp_mask(n_comp, width, s_tok.device)
targets = s_ids[:, -width:]

print("\nwidth =", width)
print("student tail tokens\n", s_tok)
print("teacher tail tokens\n", t_tok)
print("completion mask\n", mask)
print("targets (last width of student)\n", targets)

ok = True
for b in range(len(student)):
    sel = targets[b][mask[b]].tolist()
    want = student[b][-n_comp[b]:]
    if sel != want:
        ok = False
        print(f"FAIL row {b}: mask selects {sel}, answer is {want}")
    # The teacher's tail must hold the same tokens as the student's wherever the
    # mask is on; that is the alignment being asserted.
    if t_tok[b][mask[b]].tolist() != s_tok[b][mask[b]].tolist():
        ok = False
        print(f"FAIL row {b}: teacher tail {t_tok[b][mask[b]].tolist()} != "
              f"student tail {s_tok[b][mask[b]].tolist()}")

# position_ids must restart at 0 on the first real token, not on the pad.
for b in range(len(student)):
    first_real = int(s_att[b].argmax())
    if int(s_pos[b][first_real]) != 0:
        ok = False
        print(f"FAIL row {b}: position_ids start at {int(s_pos[b][first_real])}")

print("\nALIGNMENT OK" if ok else "\nALIGNMENT BROKEN")
