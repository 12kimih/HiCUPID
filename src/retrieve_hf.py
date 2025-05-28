import logging
import os
from functools import partial
from itertools import chain, islice

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, set_seed
from trl import ModelConfig, TrlParser, get_quantization_config

from .configs import HfRetrievalConfig
from .utils.logger import set_file_handler
from .utils.retrieve import build_dialogue, build_history, collate_fn_dialogue, collate_fn_question, collate_fn_utterance, compute_recall, generate, retrieve, save_recall

if __name__ == "__main__":
    parser = TrlParser((HfRetrievalConfig, ModelConfig))  # type: ignore
    configs: tuple[HfRetrievalConfig, ModelConfig] = parser.parse_args_and_config()  # type: ignore
    script_config, model_config = configs

    logger = logging.getLogger("retrieve_hf")
    logger.setLevel(logging.DEBUG)
    set_file_handler(logger, script_config.save_path)

    logger.info(f"ScriptConfig: {script_config}")
    logger.info(f"ModelConfig: {model_config}")

    load_dotenv()
    login(token=os.environ["HF_TOKEN"])

    # Preprocess data
    dialogue_dict: DatasetDict = load_dataset(script_config.dataset, name="dialogue")  # type: ignore
    dialogue: Dataset = concatenate_datasets([v.map(lambda example: {"split": k}, num_proc=script_config.num_proc) for k, v in dialogue_dict.items()])
    dialogue = Dataset.from_pandas(dialogue.to_pandas().groupby(["user_id", "dialogue_id"]).apply(build_dialogue, include_groups=False).reset_index().groupby("user_id").apply(build_history, include_groups=False).reset_index())  # type: ignore
    assert all([m["role"] == "user" for h in dialogue["messages"] for d in h for m in islice(d, 0, None, 2)])

    qa_dict: DatasetDict = load_dataset(script_config.dataset, name="qa")  # type: ignore
    qa: Dataset = concatenate_datasets([v.map(lambda example: {"split": k}, num_proc=script_config.num_proc) for k, v in qa_dict.items()])
    qa = qa.filter(lambda example: example["split"] == script_config.split, num_proc=script_config.num_proc) if script_config.split is not None else qa.filter(lambda example: "test" in example["split"], num_proc=script_config.num_proc)
    qa = qa.filter(lambda example: example["type"] == script_config.type, num_proc=script_config.num_proc) if script_config.type is not None else qa
    qa = qa.shuffle(seed=script_config.seed).select(range(script_config.num_samples)) if script_config.num_samples is not None else qa

    # Load model
    torch.set_float32_matmul_precision("high")
    if script_config.seed is not None:
        set_seed(script_config.seed)

    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=script_config.model,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        quantization_config=get_quantization_config(model_config),
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

    # Extract utterance embeddings
    dataloader = DataLoader(dialogue, batch_size=1, shuffle=False, num_workers=script_config.num_proc, collate_fn=partial(collate_fn_utterance, tokenizer=tokenizer), pin_memory=True)  # type: ignore
    embeddings = generate(model, dataloader, logger, script_config)
    dialogue = dialogue.add_column("utterance_embeddings", embeddings)  # type: ignore

    # Extract dialogue embeddings
    dataloader = DataLoader(dialogue, batch_size=1, shuffle=False, num_workers=script_config.num_proc, collate_fn=partial(collate_fn_dialogue, tokenizer=tokenizer), pin_memory=True)  # type: ignore
    embeddings = generate(model, dataloader, logger, script_config)
    dialogue = dialogue.add_column("dialogue_embeddings", embeddings)  # type: ignore

    # Extract question embeddings
    dataloader = DataLoader(qa, batch_size=script_config.batch_size, shuffle=False, num_workers=script_config.num_proc, collate_fn=partial(collate_fn_question, tokenizer=tokenizer), pin_memory=True)  # type: ignore
    embeddings = generate(model, dataloader, logger, script_config)
    qa = qa.add_column("question_embeddings", chain(*embeddings))  # type: ignore

    # Score retrieval
    qa = qa.map(retrieve, fn_kwargs={"dialogue": dialogue}, num_proc=script_config.num_proc)
    qa = qa.map(compute_recall, num_proc=script_config.num_proc)
    save_recall(qa, f"data/csv/{script_config.model.split('/')[-1]}_recall.csv")

    # Save to disk
    qa = qa.remove_columns(["question_embeddings"])
    qa.save_to_disk(script_config.save_path)

    logger.info(f"Saved dataset to {script_config.save_path}.")
