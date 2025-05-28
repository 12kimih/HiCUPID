import gc
import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "12kimih/Llama-3.2-3B-HiCUPID",
]

DATASETS = {
    "12kimih/HiCUPID": ["dialogue", "qa", "evaluation"],
}


def should_retry(exception):
    # OSError: [Errno 28] No space left on device
    if isinstance(exception, OSError) and exception.errno == 28:
        return False
    return True


def after_retry(retry_state):
    print(f"(Retry {retry_state.attempt_number:02d}/{retry_state.retry_object.stop.max_attempt_number:02d}) Exception: {retry_state.outcome.exception()}")


def load_model(repo_id):
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    del model
    del tokenizer
    gc.collect()


def load_dataset(*args, **kwargs):
    from datasets import load_dataset

    dataset = load_dataset(*args, **kwargs)
    del dataset
    gc.collect()


@retry(stop=stop_after_attempt(20), wait=wait_fixed(10), retry=retry_if_exception(should_retry), after=after_retry)
def load_model_with_retry(*args, **kwargs):
    return load_model(*args, **kwargs)


@retry(stop=stop_after_attempt(20), wait=wait_fixed(10), retry=retry_if_exception(should_retry), after=after_retry)
def load_dataset_with_retry(*args, **kwargs):
    return load_dataset(*args, **kwargs)


if __name__ == "__main__":
    load_dotenv()
    login(token=os.environ["HF_TOKEN"])

    for i, repo_id in enumerate(MODELS, start=1):
        print(f"(Model {i:02d}/{len(MODELS):02d}) Downloading {repo_id}...")
        load_model_with_retry(repo_id=repo_id)

    for i, repo_id in enumerate(DATASETS, start=1):
        subsets = DATASETS[repo_id]
        if subsets is not None:
            for j, name in enumerate(subsets, start=1):
                print(f"(Dataset {i:02d}/{len(DATASETS):02d}) (Subset {j:02d}/{len(subsets):02d}) Downloading {repo_id}/{name}...")
                load_dataset_with_retry(path=repo_id, name=name, num_proc=16)
        else:
            print(f"(Dataset {i:02d}/{len(DATASETS):02d}) Downloading {repo_id}...")
            load_dataset_with_retry(path=repo_id, num_proc=16)
