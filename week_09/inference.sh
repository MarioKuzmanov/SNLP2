#!bin/bash/sh

python inference.py \
  --model models/modernBERT/model \
  --tokenizer models/modernBERT/tokenizer \
  --save models/modernBERT/test_results.txt \
  --eval_on test