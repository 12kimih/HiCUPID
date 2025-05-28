#!/bin/bash

# Submit multiple sbatch scripts
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt zero/system
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt zero/system
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt zero/system

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt zero/user
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt zero/user
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt zero/user

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt zero/user --relevant --save_dir relevant
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt zero/user --relevant --save_dir relevant
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt zero/user --relevant --save_dir relevant

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt few/system/3
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt few/system/3
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt few/system/3

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt few/user/1
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt few/user/1
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt few/user/1

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt few/user/3
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt few/user/3
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt few/user/3

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt few/user/5
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt few/user/5
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt few/user/5

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt few/user/3 --relevant --save_dir relevant
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --prompt few/user/3 --relevant --save_dir relevant
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --prompt few/user/3 --relevant --save_dir relevant

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag bm25/utterance/5
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag bm25/utterance/15
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag bm25/utterance/25
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag bm25/dialogue/1
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag bm25/dialogue/3
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag bm25/dialogue/5

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --rag bm25/utterance/5
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --rag bm25/utterance/5

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag contriever/utterance/5
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag contriever/utterance/15
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag contriever/utterance/25
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag contriever/dialogue/1
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag contriever/dialogue/3
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --rag contriever/dialogue/5

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --rag contriever/utterance/5
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --rag contriever/utterance/5

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/lr/1e-6
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/lr/3e-6
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/lr/1e-5
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/lr/3e-5
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/lr/1e-4
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/lr/3e-4

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/rank/8
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/rank/16
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/rank/32
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/rank/64
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft sft/rank/128

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --peft sft/lr/1e-4
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --peft sft/lr/1e-4

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/lr/1e-6
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/lr/3e-6
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/lr/1e-5
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/lr/3e-5
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/lr/1e-4
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/lr/3e-4

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/rank/8
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/rank/16
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/rank/32
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/rank/64
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --peft dpo/rank/128

sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model mistralai/Mistral-7B-Instruct-v0.3 --peft dpo/lr/1e-6
sbatch scripts/slurm/inference_hf.slurm --config configs/inference_hf.yaml --model Qwen/Qwen2.5-7B-Instruct --peft dpo/lr/1e-6
