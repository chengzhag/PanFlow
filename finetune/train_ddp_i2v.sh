#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export WANDB_NAME=panflow

# Model Configuration
MODEL_ARGS=(
    --model-path "THUDM/CogVideoX-5B-I2V"
    --model-name "cogvideox-i2v"  # ["cogvideox-i2v"]
    --model-type "i2v"
    --training-type "lora"
)

# Output Configuration
OUTPUT_ARGS=(
    # --output-dir "/path/to/your/output_dir"
    # --report-to "all"
    # --backup-dir "backup"  # directory to back up the model and training state
)

# Data Configuration
DATA_ARGS=(
    # --data-root "data/360-1M"
    # --num-validation-samples "3"
    --derotation "yes"
)

# Training Configuration
TRAIN_ARGS=(
    --train-epochs 40 # number of training epochs
    # --seed 42 # random seed
    --batch-size 4
    --gradient-accumulation-steps 1
    # --mixed-precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
)

# System Configuration
SYSTEM_ARGS=(
    # --num-workers 8
    # --pin-memory
    # --nccl-timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing-steps 500 # save checkpoint every x steps
    # --checkpointing-limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume-from-checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
    # --finetune-from-checkpoint "logs/n2l07uqq/checkpoint-5000"
)

# Validation Configuration
VALIDATION_ARGS=(
    # --do-validation
    --validation-steps 500  # should be multiple of checkpointing_steps
    # --fps 16
)

# Combine all arguments and launch training
accelerate launch finetune/train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
