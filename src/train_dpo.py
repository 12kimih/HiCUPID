import os
import random
from itertools import chain

import torch
from accelerate import PartialState
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from pandas import Series
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer, ModelConfig, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config, maybe_apply_chat_template
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from .configs import ScriptArguments
from .constants import INFERENCE_PROMPTS, MISTRAL_CHAT_TEMPLATE


def build_dialogue(group):
    messages = [[{"role": "user", "content": u}, {"role": "assistant", "content": a}] for u, a in zip(group["user"], group["assistant"])]
    return Series({"type": group["type"].iloc[0], "messages": list(chain(*messages))})


def build_history(group):
    return Series({"dialogue_id": group["dialogue_id"].tolist(), "type": group["type"].tolist(), "messages": group["messages"].tolist()})


def build_messages(example, dialogue, prompt, relevant, num_exclude):
    dialogue = dialogue[example["user_id"]]
    if relevant:
        context = list(chain(*[m for i, m in zip(dialogue["dialogue_id"], dialogue["messages"]) if i in example["dialogue_id"]]))
    elif num_exclude is not None:
        excluded = random.sample([i for i, t in zip(dialogue["dialogue_id"], dialogue["type"]) if t == "persona" and i not in example["dialogue_id"]], num_exclude)
        context = list(chain(*[m for i, m in zip(dialogue["dialogue_id"], dialogue["messages"]) if i not in excluded]))
    else:
        context = list(chain(*dialogue["messages"]))
    system = [m for m in prompt if m["role"] == "system"]
    user = [m for m in prompt if m["role"] != "system"]
    question = [{"role": "user", "content": example["question"]}]
    chosen = [{"role": "assistant", "content": example["personalized_answer"]}]
    rejected = [{"role": "assistant", "content": example["general_answer"]}]
    return {"prompt": system + context + user + question, "chosen": chosen, "rejected": rejected}


def main():
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))  # type: ignore
    configs: tuple[ScriptArguments, DPOConfig, ModelConfig] = parser.parse_args_and_config()  # type: ignore
    script_args, training_args, model_config = configs
    assert model_config.model_name_or_path is not None

    load_dotenv()
    login(token=os.environ["HF_TOKEN"])
    random.seed(script_args.train_seed)

    quantization_config = get_quantization_config(model_config)
    peft_config = get_peft_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs)
    ref_model = None if peft_config is not None else AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if model_config.model_name_or_path == "mistralai/Mistral-7B-Instruct-v0.3":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool]

    with PartialState().local_main_process_first():
        dialogue: Dataset = load_dataset(script_args.dataset_name, name="dialogue", split="train")  # type: ignore
        dialogue = Dataset.from_pandas(dialogue.to_pandas().groupby(["user_id", "dialogue_id"]).apply(build_dialogue, include_groups=False).reset_index().groupby("user_id").apply(build_history, include_groups=False).reset_index())  # type: ignore
        qa: Dataset = load_dataset(script_args.dataset_name, name="qa", split="train")  # type: ignore
        qa = qa.map(build_messages, fn_kwargs={"dialogue": dialogue, "prompt": INFERENCE_PROMPTS[script_args.prompt], "relevant": script_args.relevant, "num_exclude": script_args.num_exclude}, num_proc=training_args.dataset_num_proc)
        qa = qa.map(maybe_apply_chat_template, fn_kwargs={"tokenizer": tokenizer}, num_proc=training_args.dataset_num_proc)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=qa,
        processing_class=tokenizer,
        peft_config=peft_config,  # type: ignore
    )

    trainer.accelerator.print(f"ScriptArguments: {script_args}")
    trainer.accelerator.print(f"DPOConfig: {training_args}")
    trainer.accelerator.print(f"ModelConfig: {model_config}")
    trainer.accelerator.print(f"Prompt: {qa[0]['prompt']}")
    trainer.accelerator.print(f"Chosen: {qa[0]['chosen']}")
    trainer.accelerator.print(f"Rejected: {qa[0]['rejected']}")

    if model_config.use_peft:
        trainer.accelerator.print(f"{trainer.model.peft_config}")
        trainer.model.print_trainable_parameters()  # type: ignore
        if getattr(trainer.accelerator.state, "fsdp_plugin", None):
            from peft.utils.other import fsdp_auto_wrap_policy

            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)  # type: ignore

    if script_args.train_seed is not None:
        set_seed(script_args.train_seed)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")  # type: ignore
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    main()
