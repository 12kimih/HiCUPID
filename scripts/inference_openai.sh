#!/bin/bash

# Initialize conda
__conda_setup="$($HOME/miniconda3/bin/conda shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# Activate conda environment
conda activate cupid

# Function to execute when Ctrl+C is pressed
cleanup() {
    echo "Stopping all background processes..."
    kill $(jobs -p)
    wait
    echo "All processes stopped."
}

# Use trap to catch SIGINT (Ctrl+C)
trap cleanup SIGINT

# Run multiple commands in the background
python -m src.inference_openai --config configs/inference_openai.yaml --model gpt-4o-mini-2024-07-18 --prompt zero/user &
python -m src.inference_openai --config configs/inference_openai.yaml --model gpt-4o-mini-2024-07-18 --prompt zero/user --relevant --save_dir relevant &
python -m src.inference_openai --config configs/inference_openai.yaml --model gpt-4o-mini-2024-07-18 --prompt few/user/1 &
python -m src.inference_openai --config configs/inference_openai.yaml --model gpt-4o-mini-2024-07-18 --prompt few/user/3 &
python -m src.inference_openai --config configs/inference_openai.yaml --model gpt-4o-mini-2024-07-18 --prompt few/user/5 &
python -m src.inference_openai --config configs/inference_openai.yaml --model gpt-4o-mini-2024-07-18 --prompt few/user/3 --relevant --save_dir relevant &

# Wait for all background processes to finish
wait
