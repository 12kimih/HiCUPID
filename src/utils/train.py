from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForChatML:
    # https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py
    tokenizer: PreTrainedTokenizerBase
    ignore_index: int = -100
    max_length: int | None = None
    messages_key: str = "messages"

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer does not have a pad token. Please set `pad_token_id` in the tokenizer.")
        if self.max_length is None:
            # set a sensible default
            self.max_length = min(self.tokenizer.model_max_length, 1024)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompts = []
        completions = []

        for example in examples:
            messages = example[self.messages_key]
            formatted_chat = self.tokenizer.apply_chat_template(messages, tokenize=False)

            # Split the formatted chat into prompt and completion
            assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
            last_assistant_message = assistant_messages[-1]["content"]
            prompt = formatted_chat.rsplit(last_assistant_message, 1)[0]  # type: ignore
            completion = last_assistant_message

            prompts.append(prompt)
            completions.append(completion)

        # Tokenize prompts and completions
        tokenized_prompts = self.tokenizer(prompts, truncation=True, max_length=self.max_length, padding=False, return_tensors=None)
        tokenized_completions = self.tokenizer(completions, truncation=True, max_length=self.max_length, padding=False, return_tensors=None)

        # Combine prompts and completions
        input_ids = []
        attention_mask = []
        labels = []

        for prompt, completion in zip(tokenized_prompts["input_ids"], tokenized_completions["input_ids"]):  # type: ignore
            combined_input_ids = prompt + completion
            combined_input_ids.append(self.tokenizer.eos_token_id)  # Add EOS token as final target
            combined_attention_mask = [1] * len(combined_input_ids)

            # Create labels for one-token ahead task, masking the prompt
            combined_labels = [self.ignore_index] * len(prompt) + completion
            combined_labels.append(self.tokenizer.eos_token_id)  # Add EOS token as final target

            input_ids.append(combined_input_ids)
            attention_mask.append(combined_attention_mask)
            labels.append(combined_labels)

        # first convert to list of tensors
        input_ids = [torch.tensor(ids) for ids in input_ids]
        attention_mask = [torch.tensor(mask) for mask in attention_mask]
        labels = [torch.tensor(label) for label in labels]

        # pad the input_ids, attention_mask and labels to the same length across the batch
        input_ids = pad(input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)  # type: ignore
        attention_mask = pad(attention_mask, padding_side="left", padding_value=0)
        labels = pad(labels, padding_side="left", padding_value=self.ignore_index)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    # https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output
