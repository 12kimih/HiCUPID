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
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset zero/user/GPT-4o-mini-2024-07-18 &
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset zero/user/Llama-3.1-8B-Instruct &
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset zero/user/Mistral-7B-Instruct-v0.3 &
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset zero/user/Qwen2.5-7B-Instruct &

python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset few/user/3/GPT-4o-mini-2024-07-18 &
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset few/user/3/Llama-3.1-8B-Instruct &
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset few/user/3/Mistral-7B-Instruct-v0.3 &
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset few/user/3/Qwen2.5-7B-Instruct &

python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset bm25/utterance/5/Llama-3.1-8B-Instruct &
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset bm25/utterance/5/Mistral-7B-Instruct-v0.3 &
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset bm25/utterance/5/Qwen2.5-7B-Instruct &

python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset contriever/utterance/5/Llama-3.1-8B-Instruct &
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset contriever/utterance/5/Mistral-7B-Instruct-v0.3 &
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset contriever/utterance/5/Qwen2.5-7B-Instruct &

python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset sft/lr/1e-4/Llama-3.1-8B-Instruct &
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset sft/lr/1e-4/Mistral-7B-Instruct-v0.3 &
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset sft/lr/1e-4/Qwen2.5-7B-Instruct &

python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset dpo/lr/1e-6/Llama-3.1-8B-Instruct &
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset dpo/lr/1e-6/Mistral-7B-Instruct-v0.3 &
python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset dpo/lr/1e-6/Qwen2.5-7B-Instruct &

# Wait for all background processes to finish
wait
