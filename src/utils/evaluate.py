import copy
import re

import numpy as np

from ..constants import EVALUATION_PROMPTS


def format_message(message, **kwargs):
    message["content"] = message["content"].format(**kwargs)
    return message


def build_messages(example):
    messages = copy.deepcopy(EVALUATION_PROMPTS[example["type"]])
    metadata = {k: v for subdict in example["metadata"].values() if isinstance(subdict, dict) for k, v in subdict.items()}
    question = example["question"]
    answer = example["model_answer"]
    answer_a = example["model_answer"] if example["label"] == "A" else example["personalized_answer"]
    answer_b = example["model_answer"] if example["label"] == "B" else example["personalized_answer"]
    messages = [format_message(m, **metadata, question=question, answer=answer, answer_a=answer_a, answer_b=answer_b) for m in messages]
    return {"messages": messages}


def parse_evaluation(example, idx, logger):
    parsed = True
    if example["type"] == "schedule":
        score = 0.0
        pattern = re.compile(r"Satisfaction:.*(YES|NO)", re.IGNORECASE)
        match = pattern.search(example["evaluation"])
        if match is not None:
            if match.group(1) is not None:
                score = 1.0 if match.group(1).upper() == "YES" else 0.0
            else:
                parsed = False
                logger.error(f"Parsing failed at index {idx}: {example}")
        else:
            parsed = False
            logger.error(f"Parsing failed at index {idx}: {example}")
    else:
        score = 0.5
        pattern = re.compile(r"Better Response:.*(A|B|Tie)", re.IGNORECASE)
        match = pattern.search(example["evaluation"])
        if match is not None:
            if match.group(1) is not None:
                if match.group(1).upper() == "A":
                    score = 1.0 if example["label"] == "A" else 0.0
                elif match.group(1).upper() == "B":
                    score = 1.0 if example["label"] == "B" else 0.0
            else:
                parsed = False
                logger.error(f"Parsing failed at index {idx}: {example}")
        else:
            parsed = False
            logger.error(f"Parsing failed at index {idx}: {example}")
    return {"parsed": parsed, "score": score}


def print_score(score):
    score = np.array(score)
    total = len(score)
    win = np.count_nonzero(score == 1.0) / total * 100
    tie = np.count_nonzero(score == 0.5) / total * 100
    lose = np.count_nonzero(score == 0.0) / total * 100
    return f"Score: {np.mean(score) * 100:.1f}, W/D/L: {win:.1f}/{tie:.1f}/{lose:.1f}"
