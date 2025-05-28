import os
import random
from itertools import chain

from accelerate import PartialState
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from pandas import Series
from transformers import AutoTokenizer, set_seed
from trl import ModelConfig, SFTConfig, SFTTrainer, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config

from .configs import ScriptArguments
from .constants import INFERENCE_PROMPTS
from .utils.train import DataCollatorForChatML


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
    answer = [{"role": "assistant", "content": example["personalized_answer"]}]
    return {"messages": system + context + user + question + answer}


def main():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))  # type: ignore
    configs: tuple[ScriptArguments, SFTConfig, ModelConfig] = parser.parse_args_and_config()  # type: ignore
    script_args, training_args, model_config = configs
    assert model_config.model_name_or_path is not None

    load_dotenv()
    login(token=os.environ["HF_TOKEN"])
    random.seed(script_args.train_seed)

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token

    with PartialState().local_main_process_first():
        dialogue: Dataset = load_dataset(script_args.dataset_name, name="dialogue", split="train")  # type: ignore
        dialogue = Dataset.from_pandas(dialogue.to_pandas().groupby(["user_id", "dialogue_id"]).apply(build_dialogue, include_groups=False).reset_index().groupby("user_id").apply(build_history, include_groups=False).reset_index())  # type: ignore
        qa: Dataset = load_dataset(script_args.dataset_name, name="qa", split="train")  # type: ignore
        qa = qa.map(build_messages, fn_kwargs={"dialogue": dialogue, "prompt": INFERENCE_PROMPTS[script_args.prompt], "relevant": script_args.relevant, "num_exclude": script_args.num_exclude}, num_proc=training_args.dataset_num_proc)
        data_collator = DataCollatorForChatML(tokenizer=tokenizer, max_length=training_args.max_seq_length)

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        data_collator=data_collator,
        train_dataset=qa,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.accelerator.print(f"ScriptArguments: {script_args}")
    trainer.accelerator.print(f"SFTConfig: {training_args}")
    trainer.accelerator.print(f"ModelConfig: {model_config}")
    trainer.accelerator.print(f"Messages: {qa[0]['messages']}")

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
