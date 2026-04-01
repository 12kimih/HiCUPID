# 💖 HiCUPID

We introduce 💖 **HiCUPID**, a new benchmark designed to train and evaluate Large Language Models (LLMs) for **personalized AI assistant** applications. 💖 **HiCUPID** addresses the lack of open-source conversational datasets for personalization by providing a tailored dataset and an automated evaluation model based on [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), which closely aligns with human preferences. Both the **[dataset](https://huggingface.co/datasets/12kimih/HiCUPID)** and **[evaluation model](https://huggingface.co/12kimih/Llama-3.2-3B-HiCUPID)** are available through a [HuggingFace](https://huggingface.co/12kimih) repository, with the code to reproduce the results published on [GitHub](https://github.com/12kimih/HiCUPID). For more details, please refer to our paper *"Exploring the Potential of LLMs as Personalized Assistants: Dataset, Evaluation, and Analysis,"* which has been **accepted to ACL 2025 (Main)**. You can find the paper on [arXiv](https://arxiv.org/abs/2506.01262).

---

## 📑 Table of Contents

- [💖 HiCUPID](#-hicupid)
  - [📑 Table of Contents](#-table-of-contents)
  - [🧐 Introduction](#-introduction)
  - [✨ Features](#-features)
  - [⚙️ Installation](#️-installation)
  - [🚀 Usage](#-usage)
  - [🖼️ Examples](#️-examples)
  - [⚠️ Known Issues](#️-known-issues)
  - [🙌 Contributing](#-contributing)
  - [📝 License](#-license)
  - [🔖 Citation](#-citation)

---

## 🧐 Introduction

- **Benchmark**: We present 💖 **HiCUPID**, a new benchmark specifically designed for training and evaluating Large Language Models (LLMs) as personalized assistants. 💖 **HiCUPID** is the first open-source dataset that captures the unique challenges of personalization in LLM systems.

- **Automated Evaluation Model**: 💖 **HiCUPID** includes a Llama-based automated evaluation model that assesses both the logical consistency and persona-awareness of generated responses. This proxy evaluation closely aligns with human judgment.

- **Empirical Insights**: Our extensive experiments highlight both the limitations and potential of current LLM approaches in personalized assistant tasks. The failure of existing methods underscores the difficulty of the 💖 **HiCUPID** benchmark in effectively assessing personalized LLM performance.

---

## ✨ Features

- 🧠 **Inference**: Generate answers to 💖 **HiCUPID** questions using various LLMs available on HuggingFace and OpenAI.
- ⚖️ **Evaluation**: Perform A/B evaluation by comparing model-generated answers with the ground truth.
- 🎯 **SFT/DPO**: Fine-tune LLMs for personalized AI assistants using the train split of 💖 **HiCUPID** with either [Supervised Fine-Tuning (SFT)](https://arxiv.org/abs/2106.09685) or [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290).

---

## ⚙️ Installation

To get started with this project, clone the repository and install the dependencies. **We recommend using [Miniconda](https://docs.anaconda.com/miniconda/).**

1. Clone the repository:

   ```bash
   git clone https://github.com/12kimih/HiCUPID.git
   ```

2. Navigate to the project directory:

   ```bash
   cd HiCUPID
   ```

3. Update `conda`:

   ```bash
   conda update conda -y
   ```

4. Create a new environment with Python 3.11:

   ```bash
   conda create -n cupid python=3.11 -y
   ```

5. Activate the environment:

   ```bash
   conda activate cupid
   ```

6. Install `pip` in the environment:

   ```bash
   conda install pip -y
   ```

7. Upgrade `pip`, `setuptools`, and `wheel`:

   ```bash
   pip install -U pip setuptools wheel
   ```

8. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Alternatively, you can execute steps 3 to 8 all at once by using the `env_setup.sh` script. The `env_setup.sh` script requires Miniconda to be installed at the `$HOME/miniconda3` path. Instead of steps 3 to 8, simply run the following command.

1. Run the environment setup script:

   ```bash
   ./env_setup.sh
   ```

If you have Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100), you can use [FlashAttention-2](https://github.com/Dao-AILab/flash-attention). To install the `flash-attn` package, run the following command. Note that you must have CUDA Toolkit version 11.7 or higher installed on your system.

1. Install the `flash-attn` package:

   ```bash
   pip install flash-attn --no-build-isolation
   ```

---

## 🚀 Usage

Navigate to the project directory and activate the `cupid` conda environment.

**⚠️ Important: Setup `.env` File**

Before running any code, you must create a `.env` file in the project directory containing your `HF_TOKEN` and `OPENAI_API_KEY`. Follow these steps:

1. Open the `.env` file in a text editor (e.g., `nano`, `vim`):

   ```bash
   vim .env
   ```

2. Add your `HF_TOKEN` and `OPENAI_API_KEY` in the following format:

   ```bash
   HF_TOKEN=<your_key>
   OPENAI_API_KEY=<your_key>
   ```

3. To ensure that others cannot view your credentials, change the file permissions as follows:

   ```bash
   chmod 600 .env
   ```

Make sure to complete these steps before running any part of the project. Now, you can perform various experiments using 💖 **HiCUPID**.

1. **Inference** using **HuggingFace**

   Single-GPU

   ```bash
   CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt zero/user
   ```

   Multi-GPU (2 GPUs)

   ```bash
   CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/multi_gpu.yaml --num_processes 2 -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt zero/user
   ```

   Multi-GPU (4 GPUs)

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.inference_hf --config configs/inference_hf.yaml --model meta-llama/Llama-3.1-8B-Instruct --prompt zero/user
   ```

   You can edit the default options in the provided YAML configuration file:

   ```bash
   configs/inference_hf.yaml
   ```

   You can find various command-line options in `HfInferenceConfig`:

   ```bash
   src/configs.py
   ```

2. **Inference** using **OpenAI**

   ```bash
   python -m src.inference_openai --config configs/inference_openai.yaml --model gpt-4o-mini-2024-07-18 --prompt zero/user
   ```

   You can edit the default options in the provided YAML configuration file:

   ```bash
   configs/inference_openai.yaml
   ```

   You can find various command-line options in `OpenAIInferenceConfig`:

   ```bash
   src/configs.py
   ```

3. **Evaluation** using **HuggingFace**

   Single-GPU

   ```bash
   CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset zero/user/Llama-3.1-8B-Instruct
   ```

   Multi-GPU (2 GPUs)

   ```bash
   CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/multi_gpu.yaml --num_processes 2 -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset zero/user/Llama-3.1-8B-Instruct
   ```

   Multi-GPU (4 GPUs)

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset zero/user/Llama-3.1-8B-Instruct
   ```

   You can edit the default options in the provided YAML configuration file:

   ```bash
   configs/evaluate_hf.yaml
   ```

   You can find various command-line options in `HfEvaluationConfig`:

   ```bash
   src/configs.py
   ```

4. **Evaluation** using **OpenAI**

   ```bash
   python -m src.evaluate_openai --config configs/evaluate_openai.yaml --model gpt-4o-2024-11-20 --dataset zero/user/Llama-3.1-8B-Instruct
   ```

   You can edit the default options in the provided YAML configuration file:

   ```bash
   configs/evaluate_openai.yaml
   ```

   You can find various command-line options in `OpenAIEvaluationConfig`:

   ```bash
   src/configs.py
   ```

5. **SFT**

   ```bash
   accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.train_sft --config configs/train_sft.yaml
   ```

   You can edit the default options in the provided YAML configuration file:

   ```bash
   configs/train_sft.yaml
   ```

   **SFT** runs PEFT with LoRA by default. If you want to change the hyperparameters of LoRA or modify other training arguments (e.g., `per_device_train_batch_size`, `learning_rate`), please update the `yaml` configuration file.

6. **DPO**

   ```bash
   accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m src.train_dpo --config configs/train_dpo.yaml
   ```

   You can edit the default options in the provided YAML configuration file:

   ```bash
   configs/train_dpo.yaml
   ```

   **DPO** runs PEFT with LoRA by default. If you want to change the hyperparameters of LoRA or modify other training arguments (e.g., `per_device_train_batch_size`, `learning_rate`), please update the `yaml` configuration file.

**Inference (HuggingFace), Evaluation (HuggingFace), SFT, DPO** performs distributed training (inference) in a multi-GPU environment using `accelerate`. Check out the different `accelerate` options available in the `configs/accelerate/` folder.

---

## 🖼️ Examples

1. **Inference**

   When you run the inference code, the results will be saved in `output/inference/`. You can inspect the model-generated responses as follows:

   ```python
   from datasets import Dataset

   dataset = Dataset.load_from_disk("output/inference/zero/user/Llama-3.1-8B-Instruct")

   print(f"Dataset: {dataset}")
   print(f"Question: {dataset[0]['question']}")
   print(f"Type: {dataset[0]['type']}")
   print(f"Metadata: {dataset[0]['metadata']}")
   print(f"Ground Truth: {dataset[0]['personalized_answer']}")
   print(f"Model Answer: {dataset[0]['model_answer']}")
   ```

   <details>
      <summary>Click here to see the output.</summary>

      ```text
      Dataset: Dataset({
          features: ['user_id', 'dialogue_id', 'question_id', 'question', 'personalized_answer', 'general_answer', 'type', 'metadata', 'split', 'model_answer'],
          num_rows: 20000
      })
      Question: What wildlife is unique to certain regions?
      Type: persona
      Metadata: {'persona': {'category': 'Travel', 'entity': 'New Zealand', 'relation': 'has not been to', 'sentiment': 'NEG'}, 'profile': None, 'schedule': None}
      Ground Truth: New Zealand is home to unique wildlife such as the kiwi bird, tuatara, and Hector's dolphin.
      Model Answer: You're interested in wildlife unique to certain regions. As a fan of Mark Rober's videos, you might enjoy learning about the unique wildlife found in different parts of the world. For example, the Amazon rainforest is home to over 1,300 species of birds, while the Galapagos Islands are known for their giant tortoises and marine iguanas.
      ```

   </details>

2. **Evaluation**

   You can evaluate the inference results as follows:

   ```python
   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model = AutoModelForCausalLM.from_pretrained("12kimih/Llama-3.2-3B-HiCUPID", torch_dtype=torch.bfloat16, device_map="auto")
   tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
   tokenizer.pad_token = tokenizer.eos_token

   prompt = [
       {
           "role": "user",
           "content": """
   Evaluate two responses (A and B) to a given question based on the following criteria:

   1. Personalization: Does the response effectively consider the user's provided personal information?
   2. Logical Validity: Is the response logically sound and relevant to the question?

   For each criterion, provide a brief one-line comparison of the two responses and select the better response (A, B, or Tie).

   - Ensure the comparisons are concise and directly address the criteria.
   - If both answers are equally strong or weak in a category, mark it as a Tie.
   - Do not use bold font.

   Output Format:
   1. Personalization: [Brief comparison of A and B]
   2. Logical Validity: [Brief comparison of A and B]
   Better Response: [A/B/Tie]

   Input:
   - User's Personal Information:
     - Characteristics:
       - The user has not been to New Zealand
   - Question: What wildlife is unique to certain regions?
   - Answer (A): You're interested in wildlife unique to certain regions. As a fan of Mark Rober's videos, you might enjoy learning about the unique wildlife found in different parts of the world. For example, the Amazon rainforest is home to over 1,300 species of birds, while the Galapagos Islands are known for their giant tortoises and marine iguanas.
   - Answer (B): New Zealand is home to unique wildlife such as the kiwi bird, tuatara, and Hector's dolphin.
   """.strip(),
       }
   ]

   inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, padding=True, return_dict=True, return_tensors="pt").to("cuda")
   outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
   decoded_outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].size(1) :], skip_special_tokens=True)
   print(decoded_outputs[0])
   ```

   <details>
      <summary>Click here to see the output.</summary>

      ```text
      1. Personalization: A attempts to personalize by referencing a fan of Mark Rober's videos, but this is irrelevant to the user's provided information, while B directly considers the user's lack of experience with New Zealand.
      2. Logical Validity: A provides a broader and more general answer about unique wildlife, but it is less specific and relevant to the question, whereas B is more focused and directly answers the question with examples from New Zealand.
      Better Response: B
      ```

   </details>

   When you run the evaluation code, the results will be saved in `output/evaluation/`. You can inspect the evaluation results as follows:

   ```python
   from datasets import Dataset

   dataset = Dataset.load_from_disk("output/evaluation/zero/user/Llama-3.1-8B-Instruct/Llama-3.2-3B-HiCUPID")

   print(f"Dataset: {dataset}")
   print(f"Question: {dataset[0]['question']}")
   print(f"Type: {dataset[0]['type']}")
   print(f"Metadata: {dataset[0]['metadata']}")
   print(f"Ground Truth: {dataset[0]['personalized_answer']}")
   print(f"Model Answer: {dataset[0]['model_answer']}")
   print(f"Model Label: {dataset[0]['label']}")
   print(f"Evaluation:\n{dataset[0]['evaluation']}")
   print(f"Score: {dataset[0]['score']}")
   ```

   <details>
      <summary>Click here to see the output.</summary>

      ```text
      Dataset: Dataset({
          features: ['user_id', 'dialogue_id', 'question_id', 'question', 'personalized_answer', 'general_answer', 'type', 'metadata', 'split', 'model_answer', 'label', 'evaluation', 'parsed', 'score'],
          num_rows: 20000
      })
      Question: What wildlife is unique to certain regions?
      Type: persona
      Metadata: {'persona': {'category': 'Travel', 'entity': 'New Zealand', 'relation': 'has not been to', 'sentiment': 'NEG'}, 'profile': None, 'schedule': None}
      Ground Truth: New Zealand is home to unique wildlife such as the kiwi bird, tuatara, and Hector's dolphin.
      Model Answer: You're interested in wildlife unique to certain regions. As a fan of Mark Rober's videos, you might enjoy learning about the unique wildlife found in different parts of the world. For example, the Amazon rainforest is home to over 1,300 species of birds, while the Galapagos Islands are known for their giant tortoises and marine iguanas.
      Model Label: A
      Evaluation:
      1. Personalization: A attempts to personalize by referencing a fan of Mark Rober's videos, but this is irrelevant to the user's provided information, while B directly considers the user's lack of experience with New Zealand.
      2. Logical Validity: A provides a broader and more general answer about unique wildlife, but it is less specific and relevant to the question, whereas B is more focused and directly answers the question with examples from New Zealand.
      Better Response: B
      Score: 0.0
      ```

   </details>

3. **SFT, DPO**

   When you run the model fine-tuning code, the trained LoRA adapter weights will be saved in `output/peft/`. You can inspect the fine-tuned model as follows:

   ```python
   from transformers import AutoModelForCausalLM

   model = AutoModelForCausalLM.from_pretrained("output/peft/sft/lr/1e-4/Llama-3.1-8B-Instruct")

   print(f"PEFT Config: {model.peft_config['default']}")
   ```

   <details>
      <summary>Click here to see the output.</summary>

      ```text
      PEFT Config: LoraConfig(task_type='CAUSAL_LM', peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path='meta-llama/Llama-3.1-8B-Instruct', revision=None, inference_mode=True, r=256, target_modules={'k_proj', 'gate_proj', 'up_proj', 'down_proj', 'q_proj', 'v_proj', 'o_proj'}, exclude_modules=None, lora_alpha=512, lora_dropout=0.05, fan_in_fan_out=False, bias='none', use_rslora=False, modules_to_save=None, init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', loftq_config={}, eva_config=None, use_dora=False, layer_replication=None, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=False), lora_bias=False)
      ```

   </details>

---

## ⚠️ Known Issues

1. **Inference (HuggingFace), Evaluation (HuggingFace)**
   - `bfloat16` data type may not be supported. Remove `torch_dtype=torch.bfloat16` or change it to a different data type.
   - You can use 8-bit or 4-bit quantization. For more details, please refer to [here](https://huggingface.co/docs/transformers/en/quantization/overview).
   - `flash_attention_2` may not be supported. Remove `attn_implementation="flash_attention_2"`.

2. **SFT, DPO**
   - Select the `accelerate` config file that matches the user's multi-GPU environment (e.g., number of GPUs, FSDP).
   - If GPU memory is insufficient, adjust `per_device_train_batch_size`.
   - `bfloat16` data type may not be supported. Set `torch_dtype` to `null` or `auto`.
   - You can fine-tune the model using [QLoRA](https://arxiv.org/abs/2305.14314). For more details, refer to [here](https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora).
   - `flash_attention_2` may not be supported. Set `attn_implementation` to `null`.
   - `liger_kernel` may not be supported. Set `use_liger` to `false`.

---

## 🙌 Contributing

We welcome contributions! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b my-branch`).
3. Make your changes and commit them (`git commit -m "commit message"`).
4. Push to the branch (`git push origin my-branch`).
5. Create a pull request.

Please ensure that your code follows the existing style and passes all tests before submitting a pull request.

---

## 📝 License

This project is licensed under the Apache-2.0 license - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

This work was supported by the NVIDIA Academic Grant Program.

---

## 🔖 Citation

If you use this project in your research or work, please consider citing it. Here's a suggested citation format in BibTeX:

```bibtex
@misc{mok2025exploringpotentialllmspersonalized,
      title={Exploring the Potential of LLMs as Personalized Assistants: Dataset, Evaluation, and Analysis},
      author={Jisoo Mok and Ik-hwan Kim and Sangkwon Park and Sungroh Yoon},
      year={2025},
      eprint={2506.01262},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.01262},
}
```
