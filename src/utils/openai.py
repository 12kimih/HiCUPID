import json
import os
import time
from datetime import timedelta
from itertools import islice

import numpy as np
from openai.types import Batch
from openai.types.chat import ChatCompletion
from tqdm import tqdm


def count_tokens(example, tokenizer, feature):
    tokens = 0
    for message in example[feature]:
        tokens += 3
        for k, v in message.items():
            tokens += len(tokenizer.encode(v))
            if k == "name":
                tokens += 1
    tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return {f"{feature}_tokens": tokens}


def token_stats(tokens):
    tokens = np.array(tokens)
    return f"{np.sum(tokens)}/{round(np.mean(tokens))}/{np.min(tokens)}/{np.max(tokens)}"


def build_request(example, idx, generation_config):
    request = {"custom_id": f"request-{idx}", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": example["messages"]} | generation_config}
    return {"request": request}


def save_completions(completions, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as file:
        for c in completions:
            file.write(f"{c.model_dump_json()}\n")


def load_completions(fname):
    completions = []
    with open(fname, "r") as file:
        for l in file:
            completions.append(ChatCompletion.model_validate_json(l.strip()))
    return completions


def save_batch_input(dataset, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as file:
        for r in dataset["request"]:
            file.write(f"{json.dumps(r)}\n")


def save_batch_info(batch, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as file:
        file.write(batch.to_json())


def load_batch_info(fname):
    with open(fname, "r") as file:
        batch = Batch.model_validate_json(file.read())
    return batch


def wait_batch_completion(client, batch_id, logger, sleep):
    i = 0
    while True:
        batch = client.batches.retrieve(batch_id)
        i += 1
        logger.info(f"Try {i}: {batch}")
        if batch.status == "completed":
            return batch
        elif batch.status == "in_progress":
            time.sleep(sleep)
            continue
        elif batch.status in ["validating", "finalizing", "cancelling"]:
            time.sleep(60)
            continue
        elif batch.status in ["failed", "expired", "cancelled"]:
            logger.error(f"Batch failed: {batch.status}")
            raise ValueError(f"Batch failed: {batch.status}")
        logger.error(f"Unseen batch status: {batch.status}")
        raise NotImplementedError(f"Unseen batch status: {batch.status}")


def save_batch_output(output, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as file:
        file.write(output.text)
    return fname


def load_batch_output(fname):
    with open(fname, "r") as file:
        requests = [json.loads(l) for l in file]
    requests = sorted(requests, key=lambda x: int(x["custom_id"].split("-")[1]))
    completions = [ChatCompletion.model_validate(r["response"]["body"]) for r in requests]
    return completions


def collate_fn(batch):
    return batch[0]["messages"]


def generate(rank, dataloader, client, logger, script_config, generation_config):
    fname = f"{script_config.save_path}/res/completion_{rank}.jsonl"
    completions = load_completions(fname) if os.path.exists(fname) else []
    total_steps = len(dataloader) - len(completions)

    start = time.time()
    logger.info(f"[Process {rank}] Starting generation: {total_steps} steps in total.")
    for i, messages in tqdm(enumerate(islice(dataloader, len(completions), None), start=1), total=total_steps):
        completion = client.chat.completions.create(messages=messages, **generation_config)
        completions.append(completion)

        if i % script_config.log_every == 0 or i == total_steps:
            save_completions(completions, fname)
            end = time.time()
            elapsed = end - start
            total = elapsed * (total_steps / i)
            logger.info(f"[Process {rank}] Done {i}/{total_steps} steps - {str(timedelta(seconds=int(elapsed)))}/{str(timedelta(seconds=int(total)))}.")

    return completions


def generate_batch(rank, dataset, client, logger, script_config, generation_config):
    input_fname = f"{script_config.save_path}/res/batch_input_{rank}.jsonl"
    info_fname = f"{script_config.save_path}/res/batch_info_{rank}.json"
    output_fname = f"{script_config.save_path}/res/batch_output_{rank}.jsonl"

    if os.path.exists(output_fname):
        completions = load_batch_output(output_fname)
        return completions

    if script_config.cancel:
        if os.path.exists(info_fname):
            batch = load_batch_info(info_fname)
            client.batches.cancel(batch_id=batch.id)
        return []

    if not os.path.exists(info_fname) or script_config.reload:
        dataset = dataset.map(build_request, fn_kwargs={"generation_config": generation_config}, with_indices=True, num_proc=script_config.num_proc)
        save_batch_input(dataset, input_fname)
        input = client.files.create(file=open(input_fname, "rb"), purpose="batch")
        batch = client.batches.create(input_file_id=input.id, endpoint="/v1/chat/completions", completion_window="24h")
        save_batch_info(batch, info_fname)

    batch = load_batch_info(info_fname)
    batch = wait_batch_completion(client, batch.id, logger, script_config.sleep)
    output = client.files.content(batch.output_file_id)
    save_batch_output(output, output_fname)
    completions = load_batch_output(output_fname)

    return completions


def extract_completions(completions):
    contents = [completion.choices[0].message.content for completion in completions]
    prompt_tokens = [completion.usage.prompt_tokens for completion in completions]
    completion_tokens = [completion.usage.completion_tokens for completion in completions]
    return contents, prompt_tokens, completion_tokens
