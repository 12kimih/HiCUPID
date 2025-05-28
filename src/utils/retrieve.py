import time
from collections import defaultdict
from datetime import timedelta
from itertools import chain, islice

import numpy as np
import torch
from datasets import Dataset
from pandas import Series
from tqdm import tqdm

from ..constants import NUM_RETRIEVAL


def list_to_dict(l):
    result = defaultdict(list)
    for d in l:
        for key, value in d.items():
            result[key].append(value)
    return dict(result)


def build_dialogue(group):
    messages = [[{"role": "user", "content": u}, {"role": "assistant", "content": a}] for u, a in zip(group["user"], group["assistant"])]
    return Series({"messages": list(chain(*messages))})


def build_history(group):
    return Series({"dialogue_id": group["dialogue_id"].tolist(), "messages": group["messages"].tolist()})


def collate_fn_utterance(batch, tokenizer):
    utterances = [m["content"] for m in islice(chain(*batch[0]["messages"]), 0, None, 2)]
    inputs = tokenizer(utterances, padding=True, truncation=True, return_tensors="pt")
    return inputs


def collate_fn_dialogue(batch, tokenizer):
    dialogues = [" ".join([m["content"] for m in dialogue]) for dialogue in batch[0]["messages"]]
    inputs = tokenizer(dialogues, padding=True, truncation=True, return_tensors="pt")
    return inputs


def collate_fn_question(batch, tokenizer):
    question = [example["question"] for example in batch]
    inputs = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
    return inputs


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def generate(model, dataloader, logger, script_config):
    outputs = []

    start = time.time()
    logger.info(f"Starting generation: {len(dataloader)} steps in total.")
    for i, inputs in tqdm(enumerate(dataloader, start=1), total=len(dataloader)):
        inputs = inputs.to(model.device)
        with torch.no_grad():
            embeddings = model(**inputs)
            pooled = mean_pooling(embeddings.last_hidden_state, inputs.attention_mask)
        outputs.append(pooled.tolist())

        if i % script_config.log_every == 0 or i == len(dataloader):
            end = time.time()
            elapsed = end - start
            total = elapsed * (len(dataloader) / i)
            logger.info(f"Done {i}/{len(dataloader)} steps - {str(timedelta(seconds=int(elapsed)))}/{str(timedelta(seconds=int(total)))}.")

    return outputs


def retrieve(example, dialogue):
    dialogue = dialogue[example["user_id"]]
    dialogue_id = list(chain(*[[i] * len(m) for i, m in zip(dialogue["dialogue_id"], dialogue["messages"])]))
    messages = list(chain(*dialogue["messages"]))

    # retrieve by utterance
    utterance_embeddings = dialogue["utterance_embeddings"]
    scores = torch.matmul(torch.tensor(utterance_embeddings), torch.tensor(example["question_embeddings"]))
    _, top_k_indices = torch.topk(scores, k=NUM_RETRIEVAL["utterance"][-1])
    relevant_utterances = [{"index": i, "dialogue_id": dialogue_id[2 * i], "messages": messages[2 * i : 2 * (i + 1)], "score": scores[i]} for i in top_k_indices.tolist()]

    # retrieve by dialogue
    dialogue_embeddings = dialogue["dialogue_embeddings"]
    scores = torch.matmul(torch.tensor(dialogue_embeddings), torch.tensor(example["question_embeddings"]))
    _, top_k_indices = torch.topk(scores, k=NUM_RETRIEVAL["dialogue"][-1])
    relevant_dialogues = [{"index": i, "dialogue_id": dialogue["dialogue_id"][i], "messages": dialogue["messages"][i], "score": scores[i]} for i in top_k_indices.tolist()]

    return {"relevant_utterances": relevant_utterances, "relevant_dialogues": relevant_dialogues}


def compute_recall(example):
    t = example["type"]
    utterance_recall = {str(k): {"persona": -1, "profile_1": -1, "profile_2": -1, "schedule": -1} for k in NUM_RETRIEVAL["utterance"]}
    dialogue_recall = {str(k): {"persona": -1, "profile_1": -1, "profile_2": -1, "schedule": -1} for k in NUM_RETRIEVAL["dialogue"]}
    if t in ["persona", "schedule"]:
        for k in NUM_RETRIEVAL["utterance"]:
            utterance_recall[str(k)][t] = 1 if any(dialogue_id in [u["dialogue_id"] for u in example["relevant_utterances"][:k]] for dialogue_id in example["dialogue_id"]) else 0
        for k in NUM_RETRIEVAL["dialogue"]:
            dialogue_recall[str(k)][t] = 1 if any(dialogue_id in [u["dialogue_id"] for u in example["relevant_dialogues"][:k]] for dialogue_id in example["dialogue_id"]) else 0
    elif t == "profile":
        for k in NUM_RETRIEVAL["utterance"]:
            utterance_recall[str(k)][f"{t}_1"] = 1 if any(dialogue_id in [u["dialogue_id"] for u in example["relevant_utterances"][:k]] for dialogue_id in example["dialogue_id"][:5]) else 0
            utterance_recall[str(k)][f"{t}_2"] = 1 if any(dialogue_id in [u["dialogue_id"] for u in example["relevant_utterances"][:k]] for dialogue_id in example["dialogue_id"][5:]) else 0
        for k in NUM_RETRIEVAL["dialogue"]:
            dialogue_recall[str(k)][f"{t}_1"] = 1 if any(dialogue_id in [u["dialogue_id"] for u in example["relevant_dialogues"][:k]] for dialogue_id in example["dialogue_id"][:5]) else 0
            dialogue_recall[str(k)][f"{t}_2"] = 1 if any(dialogue_id in [u["dialogue_id"] for u in example["relevant_dialogues"][:k]] for dialogue_id in example["dialogue_id"][5:]) else 0
    return {"utterance_recall": utterance_recall, "dialogue_recall": dialogue_recall}


def save_recall(dataset, path):
    rows = []
    for unit in ["dialogue", "utterance"]:
        for k in NUM_RETRIEVAL[unit]:
            recall = list_to_dict([rec[str(k)] for rec in dataset[f"{unit}_recall"]])
            recall = {key: np.mean(np.array([s for s in value if s != -1])) * 100 for key, value in recall.items()}
            persona, profile_1, profile_2, schedule = recall.values()
            profile = (profile_1 + profile_2) / 2
            total = (persona * 25 + profile * 5 + schedule * 10) / 40
            recall = {"persona": persona, "profile_1": profile_1, "profile_2": profile_2, "profile": profile, "schedule": schedule, "total": total}
            rows.append({"unit": unit, "k": k} | {key: np.round(value, 1) for key, value in recall.items()})
    Dataset.from_list(rows).to_csv(path)
