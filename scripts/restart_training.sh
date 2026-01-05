#!/bin/bash
# Script to restart training with fixed files

echo "=== Stopping old training ==="
pkill -f train_transformer.py
sleep 2

echo "=== Downloading updated files ==="
export PATH=$HOME/google-cloud-sdk/bin:$PATH
gsutil cp gs://boxing-training-data/code/train_transformer.py ~/boxing-analysis/
gsutil cp gs://boxing-training-data/code/compubox/compubox/models/adaptok_tokenizer.py ~/boxing-analysis/compubox/compubox/models/

echo "=== Starting training ==="
cd ~/boxing-analysis
nohup python3 train_transformer.py \
    --train-file ~/training_data_split/train.json \
    --val-file ~/training_data_split/val.json \
    --output-dir ~/models \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 0.001 \
    > ~/training.log 2>&1 &

echo "Training started! PID: $!"
echo "Monitor with: tail -f ~/training.log"
echo ""
echo "Current status:"
sleep 5
tail -30 ~/training.log

