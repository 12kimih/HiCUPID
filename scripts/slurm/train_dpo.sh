#!/bin/bash

# Submit multiple sbatch scripts
sbatch scripts/slurm/train_dpo.slurm --config configs/train_dpo.yaml --output_dir output/peft/dpo/lr/1e-6/Llama-3.1-8B-Instruct --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --learning_rate 1e-6
sbatch scripts/slurm/train_dpo.slurm --config configs/train_dpo.yaml --output_dir output/peft/dpo/lr/3e-6/Llama-3.1-8B-Instruct --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --learning_rate 3e-6
sbatch scripts/slurm/train_dpo.slurm --config configs/train_dpo.yaml --output_dir output/peft/dpo/lr/1e-5/Llama-3.1-8B-Instruct --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --learning_rate 1e-5
sbatch scripts/slurm/train_dpo.slurm --config configs/train_dpo.yaml --output_dir output/peft/dpo/lr/3e-5/Llama-3.1-8B-Instruct --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --learning_rate 3e-5
sbatch scripts/slurm/train_dpo.slurm --config configs/train_dpo.yaml --output_dir output/peft/dpo/lr/1e-4/Llama-3.1-8B-Instruct --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --learning_rate 1e-4
sbatch scripts/slurm/train_dpo.slurm --config configs/train_dpo.yaml --output_dir output/peft/dpo/lr/3e-4/Llama-3.1-8B-Instruct --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --learning_rate 3e-4

sbatch scripts/slurm/train_dpo.slurm --config configs/train_dpo.yaml --output_dir output/peft/dpo/rank/8/Llama-3.1-8B-Instruct --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --lora_r 8 --lora_alpha 16
sbatch scripts/slurm/train_dpo.slurm --config configs/train_dpo.yaml --output_dir output/peft/dpo/rank/16/Llama-3.1-8B-Instruct --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --lora_r 16 --lora_alpha 32
sbatch scripts/slurm/train_dpo.slurm --config configs/train_dpo.yaml --output_dir output/peft/dpo/rank/32/Llama-3.1-8B-Instruct --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --lora_r 32 --lora_alpha 64
sbatch scripts/slurm/train_dpo.slurm --config configs/train_dpo.yaml --output_dir output/peft/dpo/rank/64/Llama-3.1-8B-Instruct --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --lora_r 64 --lora_alpha 128
sbatch scripts/slurm/train_dpo.slurm --config configs/train_dpo.yaml --output_dir output/peft/dpo/rank/128/Llama-3.1-8B-Instruct --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --lora_r 128 --lora_alpha 256

sbatch scripts/slurm/train_dpo.slurm --config configs/train_dpo.yaml --output_dir output/peft/dpo/lr/1e-6/Mistral-7B-Instruct-v0.3 --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 --learning_rate 1e-6
sbatch scripts/slurm/train_dpo.slurm --config configs/train_dpo.yaml --output_dir output/peft/dpo/lr/1e-6/Qwen2.5-7B-Instruct --model_name_or_path Qwen/Qwen2.5-7B-Instruct --learning_rate 1e-6
