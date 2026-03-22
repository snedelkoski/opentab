#!/bin/bash
# Wait for generation to finish, then launch training
echo "Waiting for generate_v3 to finish..."
wait $(pgrep -f "generate_data.py")
echo "Generation done. Launching training..."

nohup uv run python train.py \
  --data data/synthetic_v3.h5 \
  --steps 200000 \
  --batch_size 4 \
  --gradient_accumulation_steps 16 \
  --max_features 50 \
  --max_train_samples 448 \
  --eval_samples 64 \
  --lr 3e-4 \
  --warmup_steps 2000 \
  --save_interval 10000 \
  --eval_interval 500 \
  --log_interval 50 \
  --output_dir checkpoints \
  > checkpoints/train_v3.log 2>&1 &

echo "Training PID: $!"
