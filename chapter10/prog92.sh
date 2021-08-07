#!/usr/bin/env bash

fairseq-interactive --path save91/checkpoint10.pt data91 < kftt-data-1.0/data/tok/kyoto-test.ja | grep '^H' | cut -f3 > 92.out