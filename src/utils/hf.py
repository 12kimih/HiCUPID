import json
import os
import time
from datetime import timedelta
from itertools import islice

import numpy as np
import torch
from tqdm import tqdm


def count_tokens(example, tokenizer, feature):
    return {f"{feature}_tokens": len(tokenizer(example[feature], add_special_tokens=False).input_ids)}


def token_stats(tokens):
    tokens = np.array(tokens)
    return f"{np.sum(tokens)}/{round(np.mean(tokens))}/{np.min(tokens)}/{np.max(tokens)}"


def build_prompt(example, tokenizer):
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt}


def collate_fn(batch, tokenizer):
    prompt = [example["prompt"] for example in batch]
    inputs = tokenizer(prompt, add_special_tokens=False, padding=True, return_tensors="pt")
    return inputs


def save_outputs(outputs, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as file:
        for o in outputs:
            file.write(f"{json.dumps(o)}\n")  # ensure_ascii=False


def load_outputs(fname):
    outputs = []
    with open(fname, "r") as file:
        for l in file:
            outputs.append(json.loads(l.strip()))
    return outputs


def gather_outputs(fname, num_processes):
    outputs = []
    for idx in range(num_processes):
        with open(fname.format(idx=idx), "r") as file:
            for l in file:
                outputs.append(json.loads(l.strip()))
    return outputs


def generate(model, dataloader, accelerator, logger, script_config, generation_config):
    fname = f"{script_config.save_path}/res/output_{accelerator.process_index}.jsonl"
    outputs = load_outputs(fname) if os.path.exists(fname) else []
    total_steps = len(dataloader) - (len(outputs) // dataloader.batch_size)

    start = time.time()
    logger.info(f"[Process {accelerator.process_index}] Starting generation: {total_steps} steps in total.")
    for i, inputs in tqdm(enumerate(islice(dataloader, len(outputs), None), start=1), total=total_steps):
        inputs = inputs.to(accelerator.device)
        with torch.no_grad():
            generated_ids = accelerator.unwrap_model(model).generate(**inputs, generation_config=generation_config)
        outputs.extend(generated_ids[:, inputs.input_ids.size(1) :].tolist())

        if i % script_config.log_every == 0 or i == total_steps:
            save_outputs(outputs, fname)
            end = time.time()
            elapsed = end - start
            total = elapsed * (total_steps / i)
            logger.info(f"[Process {accelerator.process_index}] Done {i}/{total_steps} steps - {str(timedelta(seconds=int(elapsed)))}/{str(timedelta(seconds=int(total)))}.")

    return outputs


def decode(example, tokenizer, feature):
    return {feature: tokenizer.decode(example[f"{feature}_ids"], skip_special_tokens=True)}
