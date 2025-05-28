#!/bin/bash

# environment configs
default_name="cupid"
python_version="3.11"
conda_packages=(
    "pip"
)
pip_packages=(
    "jupyter black[jupyter] isort nbqa python-dotenv gpustat tqdm tenacity protobuf packaging ninja"
    "numpy scipy pandas matplotlib seaborn scikit-learn"
    "torch torchvision torchaudio"
    "huggingface_hub[all] transformers datasets evaluate accelerate bitsandbytes peft trl liger-kernel"
    "openai tiktoken sentencepiece rank_bm25 rouge_score einops torchinfo tensorboardX"
    "flash-attn --no-build-isolation"
)

# read environment name
echo -e "A new conda environment will be created: $default_name\n"
echo -e "  - Press ENTER to confirm the environment name"
echo -e "  - Press CTRL-C to abort the environment setup"
echo -e "  - Or specify a different environment name\n"
read -p "[$default_name] >>> " name
if [ -z "$name" ]; then
    name="$default_name"
fi

# initialize conda
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

# conda update conda
conda update conda -y

# create conda environment
conda create -n "$name" python=$python_version -y
exit_status=$?
if [ $exit_status -ne 0 ]; then
    exit $exit_status
fi

# activate conda environment
conda activate "$name"

# install conda packages
for packages in "${conda_packages[@]}"; do
    conda install $packages -y
done

# install pip packages
pip install -U pip setuptools wheel
for packages in "${pip_packages[@]}"; do
    pip install -U $packages
done
