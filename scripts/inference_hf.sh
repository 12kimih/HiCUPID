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
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt zero/system &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt zero/system &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt zero/system &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt zero/user &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt zero/user &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt zero/user &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt zero/user --relevant --save_dir relevant &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt zero/user --relevant --save_dir relevant &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt zero/user --relevant --save_dir relevant &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt few/system/3 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt few/system/3 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt few/system/3 &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt few/user/1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt few/user/1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt few/user/1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt few/user/3 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt few/user/3 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt few/user/3 &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt few/user/5 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt few/user/5 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt few/user/5 &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt few/user/3 --relevant --save_dir relevant &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt few/user/3 --relevant --save_dir relevant &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt few/user/3 --relevant --save_dir relevant &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag bm25/utterance/5 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag bm25/utterance/15 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag bm25/utterance/25 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag bm25/dialogue/1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag bm25/dialogue/3 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag bm25/dialogue/5 &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --rag bm25/utterance/5 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --rag bm25/utterance/5 &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag contriever/utterance/5 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag contriever/utterance/15 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag contriever/utterance/25 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag contriever/dialogue/1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag contriever/dialogue/3 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag contriever/dialogue/5 &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --rag contriever/utterance/5 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --rag contriever/utterance/5 &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/lr/1e-6 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/lr/3e-6 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/lr/1e-5 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/lr/3e-5 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/lr/1e-4 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/lr/3e-4 &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/rank/8 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/rank/16 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/rank/32 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/rank/64 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/rank/128 &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --peft sft/lr/1e-4 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --peft sft/lr/1e-4 &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/lr/1e-6 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/lr/3e-6 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/lr/1e-5 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/lr/3e-5 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/lr/1e-4 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/lr/3e-4 &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/rank/8 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/rank/16 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/rank/32 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/rank/64 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/rank/128 &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --peft dpo/lr/1e-6 &
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --peft dpo/lr/1e-6 &

# Wait for all background processes to finish
wait
