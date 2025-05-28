from itertools import chain, islice

from rank_bm25 import BM25Okapi

from ..constants import NUM_RETRIEVAL


def retrieve(example, dialogue):
    dialogue = dialogue[example["user_id"]]
    dialogue_id = list(chain(*[[i] * len(m) for i, m in zip(dialogue["dialogue_id"], dialogue["messages"])]))
    messages = list(chain(*dialogue["messages"]))

    # retrieve by utterance
    tokenized_utterances = [m["content"].split() for m in islice(messages, 0, None, 2)]
    bm25 = BM25Okapi(tokenized_utterances)
    scores = bm25.get_scores(example["question"].split())
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: NUM_RETRIEVAL["utterance"][-1]]
    relevant_utterances = [{"index": i, "dialogue_id": dialogue_id[2 * i], "messages": messages[2 * i : 2 * (i + 1)], "score": scores[i]} for i in top_k_indices]

    # retrieve by dialogue
    tokenized_dialogues = [" ".join([m["content"] for m in dialogue]).split() for dialogue in dialogue["messages"]]
    bm25 = BM25Okapi(tokenized_dialogues)
    scores = bm25.get_scores(example["question"].split())
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: NUM_RETRIEVAL["dialogue"][-1]]
    relevant_dialogues = [{"index": i, "dialogue_id": dialogue["dialogue_id"][i], "messages": dialogue["messages"][i], "score": scores[i]} for i in top_k_indices]

    return {"relevant_utterances": relevant_utterances, "relevant_dialogues": relevant_dialogues}
