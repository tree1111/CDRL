#!/bin/bash

# Directory containing the Python script
SCRIPT_NAME="02_train_realnvp_model.py"
# SCRIPT_NAME="03-script-confoundergraph.py"

LOG_DIR="/home/adam2392/projects/logs/"

# Number of GPUs available
NUM_GPUS=7

# Change to the directory containing the script
# cd "$SCRIPT_DIR"

# Define the training seeds to match np.linspace(1, 10000, 11, dtype=int)
# training_seeds=(1 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
# Define the training seeds from 1 to 100
training_seeds=($(seq 5 6))

# Loop over the training seeds and submit a job for each seed
for i in "${!training_seeds[@]}"
do
  # TRAINING_SEED=$(expr ${training_seeds[$i]} \* 20)
  TRAINING_SEED=${training_seeds[$i]}
  
  # Calculate the GPU index to use for this job
  GPU_INDEX=$(((i % $NUM_GPUS) + 1))

  # Set the environment variable for the GPU
  export CUDA_VISIBLE_DEVICES=$GPU_INDEX

  # Construct the command to run the Python script with the current training seed
  CMD="python3 $SCRIPT_NAME --seed $TRAINING_SEED --log_dir $LOG_DIR"
  
  # Optionally, you can use a job scheduler like `nohup` to run the command in the background
  # or `&` to run the command in the background
  nohup $CMD > output_glow_max5k_${SCRIPT_NAME}_seed_${TRAINING_SEED}.log 2>&1 &

  echo $GPU_INDEX
  echo "Submitted job for training seed: $TRAINING_SEED for script: $SCRIPT_NAME"
done
