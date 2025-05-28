import os
from itertools import islice

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from trl import TrlParser

from .configs import BM25Config
from .utils.bm25 import retrieve
from .utils.retrieve import build_dialogue, build_history, compute_recall, save_recall

if __name__ == "__main__":
    parser = TrlParser(BM25Config)  # type: ignore
    script_config: BM25Config = parser.parse_args_and_config()[0]

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

    # Score retrieval
    qa = qa.map(retrieve, fn_kwargs={"dialogue": dialogue}, num_proc=script_config.num_proc)
    qa = qa.map(compute_recall, num_proc=script_config.num_proc)
    save_recall(qa, "data/csv/bm25_recall.csv")

    # Save to disk
    qa.save_to_disk(script_config.save_path)
