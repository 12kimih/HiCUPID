from itertools import chain

from pandas import Series

from ..constants import INFERENCE_PROMPTS


def build_dialogue(group):
    messages = [[{"role": "user", "content": u}, {"role": "assistant", "content": a}] for u, a in zip(group["user"], group["assistant"])]
    return Series({"messages": list(chain(*messages))})


def build_history(group):
    return Series({"dialogue_id": group["dialogue_id"].tolist(), "messages": group["messages"].tolist()})


def build_messages(example, dialogue, prompt, relevant):
    dialogue = dialogue[example["user_id"]]
    context = list(chain(*[m for i, m in zip(dialogue["dialogue_id"], dialogue["messages"]) if i in example["dialogue_id"]])) if relevant else list(chain(*dialogue["messages"]))
    system = [m for m in INFERENCE_PROMPTS[prompt] if m["role"] == "system"]
    user = [m for m in INFERENCE_PROMPTS[prompt] if m["role"] != "system"]
    question = [{"role": "user", "content": example["question"]}]
    return {"messages": system + context + user + question}


def build_messages_rag(example, prompt, unit, k):
    context = list(chain(*[unit["messages"] for unit in sorted(example[f"relevant_{unit}s"][:k], key=lambda x: x["index"])]))
    system = [m for m in INFERENCE_PROMPTS[prompt] if m["role"] == "system"]
    user = [m for m in INFERENCE_PROMPTS[prompt] if m["role"] != "system"]
    question = [{"role": "user", "content": example["question"]}]
    return {"messages": system + context + user + question}
