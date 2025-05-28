import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import chain

import tiktoken
from datasets import Dataset, disable_caching, load_from_disk
from dotenv import load_dotenv
from openai import OpenAI
from torch.utils.data import DataLoader
from trl import TrlParser

from .configs import OpenAIEvaluationConfig, OpenAIGenerationConfig, OpenAIGenerationConfigForParsing
from .utils.evaluate import build_messages, parse_evaluation, print_score
from .utils.logger import set_file_handler
from .utils.openai import collate_fn, count_tokens, extract_completions, generate, generate_batch, token_stats

if __name__ == "__main__":
    parser = TrlParser((OpenAIEvaluationConfig, OpenAIGenerationConfigForParsing))  # type: ignore
    configs: tuple[OpenAIEvaluationConfig, OpenAIGenerationConfigForParsing] = parser.parse_args_and_config()  # type: ignore
    script_config, _generation_config = configs
    generation_config = OpenAIGenerationConfig(**_generation_config.__dict__, model=script_config.model, seed=script_config.seed)

    logger = logging.getLogger("evaluate_openai")
    logger.setLevel(logging.DEBUG)
    set_file_handler(logger, script_config.save_path)

    logger.info(f"ScriptConfig: {script_config}")
    logger.info(f"GenerationConfig: {generation_config}")

    load_dotenv()

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

    logger.info(f"question: {qa[0]['question']}")
    logger.info(f"personalized_answer: {qa[0]['personalized_answer']}")
    logger.info(f"model_answer: {qa[0]['model_answer']}")
    logger.info(f"metadata: {qa[0]['metadata']}")
    logger.info(f"label: {qa[0]['label']}")

    # Load model
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tokenizer = tiktoken.encoding_for_model(script_config.model)

    # Generate evaluation
    qa = qa.map(count_tokens, fn_kwargs={"tokenizer": tokenizer, "feature": "messages"}, num_proc=script_config.num_proc)

    logger.info(f"messages: {qa[0]['messages']}")
    logger.info(f"messages tokens (sum/mean/min/max): {token_stats(qa['messages_tokens'])}")

    with ThreadPoolExecutor(max_workers=script_config.num_shards) as executor:
        script_config.num_shards = min(len(qa), script_config.num_shards)
        sharded = [qa.shard(script_config.num_shards, i, contiguous=True) for i in range(script_config.num_shards)]
        if script_config.disable_batch:
            dataloaders = [DataLoader(ds, shuffle=False, collate_fn=collate_fn, pin_memory=True) for ds in sharded]  # type: ignore
            futures = [executor.submit(partial(generate, client=client, logger=logger, script_config=script_config, generation_config=generation_config.to_dict()), rank, dataloaders[rank]) for rank in range(script_config.num_shards)]
        else:
            futures = [executor.submit(partial(generate_batch, client=client, logger=logger, script_config=script_config, generation_config=generation_config.to_dict()), rank, sharded[rank]) for rank in range(script_config.num_shards)]
        results = [f.result() for f in futures]
        contents, prompt_tokens, completion_tokens = extract_completions(list(chain(*results)))  # type: ignore

    qa = qa.add_column("evaluation", contents)  # type: ignore

    logger.info(f"evaluation: {qa[0]['evaluation']}")
    logger.info(f"prompt tokens (sum/mean/min/max): {token_stats(prompt_tokens)}")
    logger.info(f"completion tokens (sum/mean/min/max): {token_stats(completion_tokens)}")

    # Score evaluation
    qa = qa.map(parse_evaluation, with_indices=True, fn_kwargs={"logger": logger}, num_proc=script_config.num_proc)

    logger.info(print_score(qa["score"]))

    # Save to disk
    qa = qa.remove_columns(["messages", "messages_tokens"])
    qa.save_to_disk(script_config.save_path)

    logger.info(f"Saved dataset to {script_config.save_path}.")
