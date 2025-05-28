import logging
import os
import random
from datetime import timedelta
from functools import partial

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import Dataset, disable_caching, load_from_disk
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
from trl import ModelConfig, TrlParser, get_quantization_config

from .configs import HfEvaluationConfig, HfGenerationConfig
from .utils.evaluate import build_messages, parse_evaluation, print_score
from .utils.hf import build_prompt, collate_fn, count_tokens, decode, gather_outputs, generate, token_stats
from .utils.logger import set_file_handler

if __name__ == "__main__":
    parser = TrlParser((HfEvaluationConfig, ModelConfig, HfGenerationConfig))  # type: ignore
    configs: tuple[HfEvaluationConfig, ModelConfig, HfGenerationConfig] = parser.parse_args_and_config()  # type: ignore
    script_config, model_config, _generation_config = configs

    # Initialize the accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=3))
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    logger = logging.getLogger("evaluate_hf")
    logger.setLevel(logging.DEBUG)
    set_file_handler(logger, script_config.save_path)

    if accelerator.is_main_process:
        logger.info(f"ScriptConfig: {script_config}")
        logger.info(f"ModelConfig: {model_config}")

    load_dotenv()
    login(token=os.environ["HF_TOKEN"])

    # Preprocess data
    disable_caching()
    random.seed(script_config.seed)

    qa: Dataset = load_from_disk(script_config.load_path)  # type: ignore
    qa = qa.filter(lambda example: example["split"] == script_config.split, num_proc=script_config.num_proc) if script_config.split is not None else qa.filter(lambda example: "test" in example["split"], num_proc=script_config.num_proc)
    qa = qa.filter(lambda example: example["type"] == script_config.type, num_proc=script_config.num_proc) if script_config.type is not None else qa
    qa = qa.shuffle(seed=script_config.seed).select(range(script_config.num_samples)) if script_config.num_samples is not None else qa

    num_samples = len(qa)
    label = ["A"] * (num_samples // 2) + ["B"] * (num_samples - num_samples // 2)
    random.shuffle(label)
    qa = qa.add_column("label", label)  # type: ignore
    qa = qa.map(build_messages, num_proc=script_config.num_proc)

    if accelerator.is_main_process:
        logger.info(f"question: {qa[0]['question']}")
        logger.info(f"personalized_answer: {qa[0]['personalized_answer']}")
        logger.info(f"model_answer: {qa[0]['model_answer']}")
        logger.info(f"metadata: {qa[0]['metadata']}")
        logger.info(f"label: {qa[0]['label']}")

    # Load model
    torch.set_float32_matmul_precision("high")
    if script_config.seed is not None:
        set_seed(script_config.seed)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=script_config.model,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        quantization_config=get_quantization_config(model_config),
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    generation_config = GenerationConfig.from_pretrained(model.config._name_or_path)
    unused_kwargs = generation_config.update(pad_token_id=tokenizer.pad_token_id, **_generation_config.to_dict())

    if accelerator.is_main_process:
        logger.info(f"GenerationConfig: {generation_config}")
        if unused_kwargs:
            logger.warning(f"Unused GenerationConfig: {unused_kwargs}")

    # Generate evaluation
    qa = qa.map(build_prompt, fn_kwargs={"tokenizer": tokenizer}, num_proc=script_config.num_proc)
    qa = qa.map(count_tokens, fn_kwargs={"tokenizer": tokenizer, "feature": "prompt"}, num_proc=script_config.num_proc)

    if accelerator.is_main_process:
        logger.info(f"prompt: {qa[0]['prompt']}")
        logger.info(f"prompt tokens (sum/mean/min/max): {token_stats(qa['prompt_tokens'])}")

    model = accelerator.prepare(model)
    shard = qa.shard(accelerator.num_processes, accelerator.process_index, contiguous=True)
    dataloader = DataLoader(shard, batch_size=script_config.batch_size, shuffle=False, num_workers=script_config.num_proc, collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True)  # type: ignore
    output_ids = generate(model, dataloader, accelerator, logger, script_config, generation_config)

    # Ensure all processes have completed saving their outputs
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        output_ids = gather_outputs(f"{script_config.save_path}/res/output_{{idx}}.jsonl", accelerator.num_processes)

        qa = qa.add_column("evaluation_ids", output_ids)  # type: ignore
        qa = qa.map(decode, fn_kwargs={"tokenizer": tokenizer, "feature": "evaluation"}, num_proc=script_config.num_proc)

        logger.info(f"evaluation: {qa[0]['evaluation']}")

        # Score evaluation
        qa = qa.map(parse_evaluation, with_indices=True, fn_kwargs={"logger": logger}, num_proc=script_config.num_proc)

        logger.info(print_score(qa["score"]))

        # Save to disk
        qa = qa.remove_columns(["messages", "prompt", "prompt_tokens", "evaluation_ids"])
        qa.save_to_disk(script_config.save_path)

        logger.info(f"Saved dataset to {script_config.save_path}.")
