from dataclasses import asdict, dataclass, field

from .constants import MODEL_NAME


@dataclass
class HfInferenceConfig:
    model: str = field(default="meta-llama/Llama-3.1-8B-Instruct", metadata={"help": "Model name."})
    dataset: str = field(default="12kimih/HiCUPID", metadata={"help": "Dataset name."})
    prompt: str = field(default="zero/system", metadata={"help": "Prompt name (e.g., {zero,few}/{system,user}[/{n}])."})
    relevant: bool = field(default=False, metadata={"help": "Whether to include only relevant dialogues."})
    peft: str | None = field(default=None, metadata={"help": "PEFT configuration (e.g., {sft,dpo}/{rank,lr,epoch}/{n})."})
    rag: str | None = field(default=None, metadata={"help": "RAG configuration (e.g., {bm25,contriever}/{utterance,dialogue}/{n})."})
    batch_size: int = field(default=2, metadata={"help": "Batch size."})
    num_proc: int = field(default=16, metadata={"help": "Number of workers to use for dataset processing."})
    log_every: int = field(default=10, metadata={"help": "Interval for logging, specified in steps."})
    split: str | None = field(default=None, metadata={"help": "Which test split to use."})
    type: str | None = field(default=None, metadata={"help": "Which type of questions to use (e.g., {persona,profile,schedule})."})
    num_samples: int | None = field(default=None, metadata={"help": "Number of samples to select."})
    peft_dir: str | None = field(default=None, metadata={"help": "Name of the directory to load the peft adapters."})
    rag_dir: str | None = field(default=None, metadata={"help": "Name of the directory to load the retrievals."})
    save_dir: str | None = field(default=None, metadata={"help": "Name of the directory to save the outputs."})
    seed: int | None = field(default=None, metadata={"help": "Seed number for random number generation."})

    def __post_init__(self):
        self.model_path = "/".join(["output", self.peft_dir or "peft"] + self.peft.split("/") + self.model.split("/")[-1:]) if self.peft is not None else None
        self.dataset_path = "/".join(["output", self.rag_dir or "rag"] + self.rag.split("/")[:1]) if self.rag is not None else None
        self.save_path = "/".join(["output", self.save_dir or "inference"] + (self.peft or self.rag or self.prompt).split("/") + self.model.split("/")[-1:])


@dataclass
class HfEvaluationConfig:
    model: str = field(default="12kimih/Llama-3.2-3B-HiCUPID", metadata={"help": "Model name or path to a local model checkpoint."})
    dataset: str = field(default="zero/system/Llama-3.1-8B-Instruct", metadata={"help": "Path to the dataset to evaluate (e.g., {zero,few|sft,dpo|bm25,contriever}/{system,user|rank,lr,epoch|utterance,dialogue}[/{n}]/{model})."})
    batch_size: int = field(default=64, metadata={"help": "Batch size."})
    num_proc: int = field(default=16, metadata={"help": "Number of workers to use for dataset processing."})
    log_every: int = field(default=10, metadata={"help": "Interval for logging, specified in steps."})
    split: str | None = field(default=None, metadata={"help": "Which test split to use."})
    type: str | None = field(default=None, metadata={"help": "Which type of questions to use. (e.g., {persona,profile,schedule})"})
    num_samples: int | None = field(default=None, metadata={"help": "Number of samples to select."})
    load_dir: str | None = field(default=None, metadata={"help": "Name of the directory to load the dataset."})
    save_dir: str | None = field(default=None, metadata={"help": "Name of the directory to save the outputs."})
    seed: int | None = field(default=None, metadata={"help": "Seed number for random number generation."})

    def __post_init__(self):
        self.load_path = "/".join(["output", self.load_dir or "inference"] + self.dataset.split("/"))
        self.save_path = "/".join(["output", self.save_dir or "evaluation"] + self.dataset.split("/") + self.model.split("/")[-1:])


@dataclass
class HfRetrievalConfig:
    model: str = field(default="facebook/contriever", metadata={"help": "Model name."})
    dataset: str = field(default="12kimih/HiCUPID", metadata={"help": "Dataset name."})
    batch_size: int = field(default=256, metadata={"help": "Batch size."})
    num_proc: int = field(default=16, metadata={"help": "Number of workers to use for dataset processing."})
    log_every: int = field(default=10, metadata={"help": "Interval for logging, specified in steps."})
    split: str | None = field(default=None, metadata={"help": "Which test split to use."})
    type: str | None = field(default=None, metadata={"help": "Which type of questions to use. (e.g., {persona,profile,schedule})"})
    num_samples: int | None = field(default=None, metadata={"help": "Number of samples to select."})
    save_dir: str | None = field(default=None, metadata={"help": "Name of the directory to save the outputs."})
    seed: int | None = field(default=None, metadata={"help": "Seed number for random number generation."})

    def __post_init__(self):
        self.save_path = "/".join(["output", self.save_dir or "rag", self.model.split("/")[-1]])


@dataclass
class HfGenerationConfig:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py
    # Parameters that control the length of the output
    max_length: int | None = field(default=None, metadata={"help": "The maximum length the generated tokens can have. Corresponds to the length of the input prompt + `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set."})
    max_new_tokens: int | None = field(default=None, metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."})
    min_length: int | None = field(default=None, metadata={"help": "The minimum length of the sequence to be generated. Corresponds to the length of the input prompt + `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set."})
    min_new_tokens: int | None = field(default=None, metadata={"help": "The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt."})
    early_stopping: bool | None = field(default=None, metadata={"help": 'Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values: `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates; `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).'})  # bool | str | None
    max_time: float | None = field(default=None, metadata={"help": "The maximum amount of time you allow the computation to run for in seconds. generation will still finish the current pass after allocated time has been passed."})
    stop_strings: list[str] | None = field(default=None, metadata={"help": "A string or a list of strings that should terminate generation if the model outputs them."})  # str | list[str] | None

    # Parameters that control the generation strategy used
    do_sample: bool | None = field(default=None, metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise."})
    num_beams: int | None = field(default=None, metadata={"help": "Number of beams for beam search. 1 means no beam search."})
    num_beam_groups: int | None = field(default=None, metadata={"help": "Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details."})
    penalty_alpha: float | None = field(default=None, metadata={"help": "The values balance the model confidence and the degeneration penalty in contrastive search decoding."})
    dola_layers: str | None = field(default=None, metadata={"help": 'The layers to use for DoLa decoding. If `None`, DoLa decoding is not used. If a string, it must be one of "low" or "high", which means using the lower part or higher part of the model layers, respectively. "low" means the first half of the layers up to the first 20 layers, and "high" means the last half of the layers up to the last 20 layers. If a list of integers, it must contain the indices of the layers to use for candidate premature layers in DoLa. The 0-th layer is the word embedding layer of the model. Set to `\'low\'` to improve long-answer reasoning tasks, `\'high\'` to improve short-answer tasks. Check the [documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/generation_strategies.md) or [the paper](https://arxiv.org/abs/2309.03883) for more details.'})  # str | list[int] | None

    # Parameters for manipulation of the model output logits
    temperature: float | None = field(default=None, metadata={"help": "The value used to modulate the next token probabilities."})
    top_k: int | None = field(default=None, metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."})
    top_p: float | None = field(default=None, metadata={"help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation."})
    min_p: float | None = field(default=None, metadata={"help": "Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0 and 1. Typical values are in the 0.01-0.2 range, comparably selective as setting `top_p` in the 0.99-0.8 range (use the opposite of normal `top_p` values)."})
    typical_p: float | None = field(default=None, metadata={"help": "Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to `typical_p` or higher are kept for generation. See [this paper](https://arxiv.org/pdf/2202.00666.pdf) for more details."})
    epsilon_cutoff: float | None = field(default=None, metadata={"help": "If set to float strictly between 0 and 1, only tokens with a conditional probability greater than `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the size of the model. See [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more details."})
    eta_cutoff: float | None = field(default=None, metadata={"help": "Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between 0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits)))`. The latter term is intuitively the expected next token probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model. See [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more details."})
    diversity_penalty: float | None = field(default=None, metadata={"help": "This value is subtracted from a beam's score if it generates a token same as any beam from other group at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is enabled."})
    repetition_penalty: float | None = field(default=None, metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details."})
    encoder_repetition_penalty: float | None = field(default=None, metadata={"help": "The paramater for encoder_repetition_penalty. An exponential penalty on sequences that are not in the original input. 1.0 means no penalty."})
    length_penalty: float | None = field(default=None, metadata={"help": "Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences."})
    no_repeat_ngram_size: int | None = field(default=None, metadata={"help": "If set to int > 0, all ngrams of that size can only occur once."})

    def to_dict(self):
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class OpenAIInferenceConfig:
    model: str = field(default="gpt-4o-mini", metadata={"help": "Model name."})
    dataset: str = field(default="12kimih/HiCUPID", metadata={"help": "Dataset name."})
    prompt: str = field(default="zero/system", metadata={"help": "Prompt name (e.g., {zero,few}/{system,user})."})
    relevant: bool = field(default=False, metadata={"help": "Whether to include only relevant dialogues."})
    rag: str | None = field(default=None, metadata={"help": "RAG configuration (e.g., {bm25,contriever}/{utterance,dialogue}/{n})."})
    num_shards: int = field(default=64, metadata={"help": "Number of shards for multithread processing."})
    num_proc: int = field(default=16, metadata={"help": "Number of workers to use for dataset processing."})
    log_every: int = field(default=10, metadata={"help": "Interval for logging, specified in steps."})
    sleep: int = field(default=180, metadata={"help": "Time in seconds to sleep between batch retrievals."})
    cancel: bool = field(default=False, metadata={"help": "Whether to cancel submitted batch jobs."})
    reload: bool = field(default=False, metadata={"help": "Whether to ignore the existing batch and create a new one."})
    disable_batch: bool = field(default=False, metadata={"help": "Whether to disable batch API calls."})
    split: str | None = field(default=None, metadata={"help": "Which test split to use."})
    type: str | None = field(default=None, metadata={"help": "Which type of questions to use. (e.g., {persona,profile,schedule})"})
    num_samples: int | None = field(default=None, metadata={"help": "Number of samples to select."})
    rag_dir: str | None = field(default=None, metadata={"help": "Name of the directory to load the retrievals."})
    save_dir: str | None = field(default=None, metadata={"help": "Name of the directory to save the outputs."})
    seed: int | None = field(default=None, metadata={"help": "Seed number for random number generation."})

    def __post_init__(self):
        self.dataset_path = "/".join(["output", self.rag_dir or "rag"] + self.rag.split("/")[:1]) if self.rag is not None else None
        self.save_path = "/".join(["output", self.save_dir or "inference"] + (self.rag or self.prompt).split("/") + [MODEL_NAME.get(self.model, self.model)])


@dataclass
class OpenAIEvaluationConfig:
    model: str = field(default="gpt-4o", metadata={"help": "Model name or path to a local model checkpoint."})
    dataset: str = field(default="zero/system/Llama-3.1-8B-Instruct", metadata={"help": "Path to the dataset to evaluate (e.g., {zero,few|sft,dpo|bm25,contriever}/{system,user|rank,lr,epoch|utterance,dialogue}[/{n}]/{model})."})
    num_shards: int = field(default=64, metadata={"help": "Number of shards for multithread processing."})
    num_proc: int = field(default=16, metadata={"help": "Number of workers to use for dataset processing."})
    log_every: int = field(default=10, metadata={"help": "Interval for logging, specified in steps."})
    sleep: int = field(default=180, metadata={"help": "Time in seconds to sleep between batch retrievals."})
    cancel: bool = field(default=False, metadata={"help": "Whether to cancel submitted batch jobs."})
    reload: bool = field(default=False, metadata={"help": "Whether to ignore the existing batch and create a new one."})
    disable_batch: bool = field(default=False, metadata={"help": "Whether to disable batch API calls."})
    split: str | None = field(default=None, metadata={"help": "Which test split to use."})
    type: str | None = field(default=None, metadata={"help": "Which type of questions to use. (e.g., {persona,profile,schedule})"})
    num_samples: int | None = field(default=None, metadata={"help": "Number of samples to select."})
    load_dir: str | None = field(default=None, metadata={"help": "Name of the directory to load the dataset."})
    save_dir: str | None = field(default=None, metadata={"help": "Name of the directory to save the outputs."})
    seed: int | None = field(default=None, metadata={"help": "Seed number for random number generation."})

    def __post_init__(self):
        self.load_path = "/".join(["output", self.load_dir or "inference"] + self.dataset.split("/"))
        self.save_path = "/".join(["output", self.save_dir or "evaluation"] + self.dataset.split("/") + [MODEL_NAME.get(self.model, self.model)])


@dataclass
class OpenAIGenerationConfigForParsing:
    # https://platform.openai.com/docs/api-reference/chat/create
    max_completion_tokens: int | None = field(default=None, metadata={"help": "An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens."})
    stop: list[str] | None = field(default=None, metadata={"help": "Up to 4 sequences where the API will stop generating further tokens."})  # str | list[str] | None
    temperature: float | None = field(default=None, metadata={"help": "What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both."})
    top_p: float | None = field(default=None, metadata={"help": "An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both."})
    frequency_penalty: float | None = field(default=None, metadata={"help": "Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim."})
    presence_penalty: float | None = field(default=None, metadata={"help": "Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics."})


@dataclass
class OpenAIGenerationConfig(OpenAIGenerationConfigForParsing):
    # https://platform.openai.com/docs/api-reference/chat/create
    model: str | None = field(default=None, metadata={"help": "ID of the model to use. See the model endpoint compatibility table for details on which models work with the Chat API."})
    seed: int | None = field(default=None, metadata={"help": "This feature is in Beta. If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same seed and parameters should return the same result. Determinism is not guaranteed, and you should refer to the system_fingerprint response parameter to monitor changes in the backend."})

    def to_dict(self):
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class BM25Config:
    dataset: str = field(default="12kimih/HiCUPID", metadata={"help": "Dataset name."})
    num_proc: int = field(default=64, metadata={"help": "Number of workers to use for dataset processing."})
    split: str | None = field(default=None, metadata={"help": "Which test split to use."})
    type: str | None = field(default=None, metadata={"help": "Which type of questions to use. (e.g., {persona,profile,schedule})"})
    num_samples: int | None = field(default=None, metadata={"help": "Number of samples to select."})
    save_dir: str | None = field(default=None, metadata={"help": "Name of the directory to save the outputs."})
    seed: int | None = field(default=None, metadata={"help": "Seed number for random number generation."})

    def __post_init__(self):
        self.save_path = "/".join(["output", self.save_dir or "rag", "bm25"])


@dataclass
class BleuRougeConfig:
    dataset: str = field(default="zero/system/Llama-3.1-8B-Instruct", metadata={"help": "Path to the dataset to evaluate (e.g., {zero,few|sft,dpo|bm25,contriever}/{system,user|rank,lr,epoch|utterance,dialogue}[/{n}]/{model})."})
    num_proc: int = field(default=64, metadata={"help": "Number of workers to use for dataset processing."})
    split: str | None = field(default=None, metadata={"help": "Which test split to use."})
    type: str | None = field(default=None, metadata={"help": "Which type of questions to use. (e.g., {persona,profile,schedule})"})
    num_samples: int | None = field(default=None, metadata={"help": "Number of samples to select."})
    load_dir: str | None = field(default=None, metadata={"help": "Name of the directory to load the dataset."})
    save_dir: str | None = field(default=None, metadata={"help": "Name of the directory to save the outputs."})
    seed: int | None = field(default=None, metadata={"help": "Seed number for random number generation."})

    def __post_init__(self):
        self.load_path = "/".join(["output", self.load_dir or "inference"] + self.dataset.split("/"))
        self.save_path = "/".join(["output", self.save_dir or "evaluation", "bleu_rouge"] + self.dataset.split("/"))


@dataclass
class ScriptArguments:
    # https://github.com/huggingface/trl/blob/main/trl/utils.py
    dataset_name: str = field(metadata={"help": "Dataset name."})
    dataset_config: str | None = field(default=None, metadata={"help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` function."})
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})
    gradient_checkpointing_use_reentrant: bool = field(default=False, metadata={"help": "Whether to apply `use_reentrant` for gradient_checkpointing."})
    ignore_bias_buffers: bool = field(default=False, metadata={"help": "Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar type, inplace operation. See https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992."})

    prompt: str = field(default="zero/system", metadata={"help": "Prompt name (e.g., {zero,few}/{system,user})."})
    relevant: bool = field(default=False, metadata={"help": "Whether to include only relevant dialogues."})
    num_exclude: int | None = field(default=None, metadata={"help": "How many persona dialogues to exclude; relevant dialogues are always included."})
    margin: float | None = field(default=None, metadata={"help": "Loss margin between chosen and rejected for reward modeling."})
    train_seed: int | None = field(default=None, metadata={"help": "Seed number for training."})
