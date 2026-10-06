#!/bin/sh
# Runs every step in order and prints what the video shows. About four minutes the first time.
set -e
cd "$(dirname "$0")"
for f in step1_base step2_full_finetune step3_the_change step4_low_rank step5_lora_layer step6_train_lora step7_merge_swap step8_rank exercise; do
  echo "== $f"
  python3 $f.py
done
