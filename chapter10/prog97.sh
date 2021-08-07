#!/usr/bin/env bash

export MKL_THREADING_LAYER="GNU"

for dropout in 0.1 0.3 0.5
do
  for lr in 1e-4 1e-3 1e-2 1e-1
  do
    fairseq-train data91 \
    --fp16 \
    --save-dir save97_"$dropout"_"$lr" \
    --max-epoch 10 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr "$lr" --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --dropout "$dropout" --weight-decay 0.0001 \
    --update-freq 1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 97_"$dropout"_"$lr".log

    fairseq-interactive --path save97_"$dropout"_"$lr"/checkpoint10.pt data91 < kftt-data-1.0/data/tok/kyoto-test.ja | grep '^H' | cut -f3 > 97_"$dropout"_"$lr".out

    fairseq-score --sys 97_"$dropout"_"$lr".out --ref kftt-data-1.0/data/tok/kyoto-test.en
  done
done