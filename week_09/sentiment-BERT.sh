#!bin/bash/sh

python sentiment-BERT.py \
      --model_name_or_path FacebookAI/roberta-base \
      --max_length 100 \
      --output_dir models/roberta-base \
      --batch_size 32 \
      --epochs 3 \
      --lr 5e-5