#!/bin/bash
#SBATCH --job-name=cupid                   # Name of the job, used in job scheduling and output filenames
#SBATCH --output=logs/%x_%j.log            # File to which stdout and stderr will be written; %x is job name, %j is job ID
#SBATCH --open-mode=append                 # Append to the output file instead of overwriting it
#SBATCH --nodes=1                          # Number of nodes to allocate for the job
#SBATCH --ntasks-per-node=1                # Number of tasks to run per node
#SBATCH --cpus-per-task=16                 # Number of CPUs allocated per task
#SBATCH --gres=gpu:1                       # Number of GPUs to allocate (4 GPUs in this case)

# Export environment variables
export MASTER_PORT=$(shuf -i 30000-40000 -n 1)
export OMP_NUM_THREADS=16

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

# Run scripts
srun accelerate launch --config_file configs/accelerate/single_gpu.yaml --main_process_port $MASTER_PORT -m src.evaluate_hf "$@"
