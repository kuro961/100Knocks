#!/usr/bin/env bash

fairseq-preprocess -s ja -t en \
--trainpref kftt-data-1.0/data/tok/kyoto-train.cln \
--validpref kftt-data-1.0/data/tok/kyoto-dev \
--destdir data91 \
--thresholdsrc 5 \
--thresholdtgt 5 \
--workers 20

export MKL_THREADING_LAYER="GNU"

fairseq-train data91 \
--fp16 \
--save-dir save91 \
--max-epoch 10 \
--arch transformer --share-decoder-input-output-embed \
--optimizer adam --clip-norm 1.0 \
--lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
--update-freq 1 \
--dropout 0.2 --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 8000 > 91.log