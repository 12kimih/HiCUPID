import evaluate
from datasets import Dataset, disable_caching, load_from_disk
from trl import TrlParser

from .configs import BleuRougeConfig


def compute_metric(example, bleu, rouge):
    try:
        bleu = bleu.compute(predictions=[example["model_answer"]], references=[[example["personalized_answer"]]], smooth=True)["bleu"]
    except Exception as e:
        print(f"Failed BLEU computation: {e}")
        bleu = 0.0
    try:
        rouge = rouge.compute(predictions=[example["model_answer"]], references=[example["personalized_answer"]])["rougeL"]
    except Exception as e:
        print(f"Failed ROUGE computation: {e}")
        rouge = 0.0
    return {"bleu": bleu, "rouge": rouge}


if __name__ == "__main__":
    parser = TrlParser(BleuRougeConfig)  # type: ignore
    script_config: BleuRougeConfig = parser.parse_args_and_config()[0]

    disable_caching()

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    qa: Dataset = load_from_disk(script_config.load_path)  # type: ignore
    qa = qa.filter(lambda example: example["split"] == script_config.split, num_proc=script_config.num_proc) if script_config.split is not None else qa.filter(lambda example: "test" in example["split"], num_proc=script_config.num_proc)
    qa = qa.filter(lambda example: example["type"] == script_config.type, num_proc=script_config.num_proc) if script_config.type is not None else qa
    qa = qa.shuffle(seed=script_config.seed).select(range(script_config.num_samples)) if script_config.num_samples is not None else qa
    qa = qa.map(compute_metric, fn_kwargs={"bleu": bleu, "rouge": rouge}, num_proc=script_config.num_proc)
    qa.save_to_disk(script_config.save_path)
