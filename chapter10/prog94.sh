#!/usr/bin/env bash

for i in $(seq 30)
do
  fairseq-interactive --path save91/checkpoint10.pt --beam "$i" data91 < kftt-data-1.0/data/tok/kyoto-test.ja | grep '^H' | cut -f3 > 94."$i".out
done

for i in $(seq 30)
do
  fairseq-score --sys 94."$i".out --ref kftt-data-1.0/data/tok/kyoto-test.en > 94."$i".score
done