#!/usr/bin/env bash
# Print the checkpoint with the highest first-letter score among the finished stages (vd, vd2, cm, num, beh, restudy*).
cd ~/projects/teleqna/runs/teleqna-8b; best=0; path=models/kit/vd/ep1
for spec in "vd:models/kit/vd/ep1" "vd2:models/kit/vd2/ep1" "cm:models/kit/cm/ep1" "num:models/kit/num/ep1" "beh:models/kit/beh/ep1" "vd3:models/kit/vd3/ep1" "cm_mlp:models/kit/cm_mlp/ep1" "cm2:models/kit/cm2/ep1" "vd4:models/kit/vd4/ep1" "dpo:models/kit/dpo/ep1" "dpo2:models/kit/dpo2/ep1" "dpo3:models/kit/dpo3/ep1" "style:models/kit/style/ep1" "vd5:models/kit/vd5/ep1" "big3_ep1:models/kit/big3/ep1" "vd8:models/kit/vd8/ep1" "big3_soup:models/kit/big3_soup" "merge_plain:models/kit/merge_plain" "merge_a:models/kit/merge_a" "merge_b:models/kit/merge_b" "merge_c:models/kit/merge_c" "ens:models/kit/ens/ep1" "ens2:models/kit/ens2/ep1" "merge_d:models/kit/merge_d" "merge_e:models/kit/merge_e" "merge_f:models/kit/merge_f" "merge_g:models/kit/merge_g" "merge_h:models/kit/merge_h" "merge_i:models/kit/merge_i" "big4_soup:models/kit/big4_soup" "vd9:models/kit/vd9/ep1" "ens3:models/kit/ens3/ep1" "merge_j:models/kit/merge_j" "merge_k:models/kit/merge_k" "merge_l:models/kit/merge_l" "utr:models/kit/utr/ep1" "het:models/kit/het/ep1" "soup_het:models/kit/soup_het" "wise_u3:models/kit/wise_u3" "sib:models/kit/sib/ep1" "vd6:models/kit/vd6/ep1" "recall:models/kit/recall/ep1" "pit:models/kit/pit/ep1"; do
  t=${spec%%:*}; p=${spec##*:}; f=logs/eval_$t.log; [ -f "$f" ] && [ -d "$p" ] || continue
  s=$(grep RESULT "$f" | sed -E "s/.*first-letter ([0-9.]+) .*/\1/"); [ -n "$s" ] || continue
  if [ "$(echo "$s $best" | awk '{print ($1 > $2)}')" = 1 ]; then best=$s; path=$p; fi
done
echo "$path $best"
