from dataclasses import dataclass, field
import math
import os
import random
from typing import Optional
from datasets import Dataset
from loguru import logger
import numpy as np
import torch
from transformers import (
    Qwen2Config,
    Qwen2Tokenizer,
    Qwen2ForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
)
from transformers.trainer import TRAINING_ARGS_NAME

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:
    from transformers.deepspeed import is_deepspeed_zero3_enabled

from dataset import SupervisedDataset, IGNORE_INDEX, get_conv_template
from typing import List
from tqdm import tqdm


@dataclass
class ModelArgs:
    """
    Model args for a GptModel
    """

    model_class: str = "ModelArgs"
    dataset_class: Dataset = None
    learning_rate: float = 2e-5
    manual_seed: int = 42
    fp16: bool = False
    bf16: bool = False
    int8: bool = False
    int4: bool = False
    debug: bool = False
    max_seq_length: int = 256  # max length of input sequence
    max_length: int = 256  # max length of the sequence to be generated
    warmup_steps: int = 50
    report_to = "tensorboard"
    optimizer: str = "adamw_torch"
    save_strategy: str = "steps"
    eval_steps: int = 200
    save_steps: int = 400
    max_eval_samples: int = 20
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    do_sample: bool = True
    temperature: float = 0.1
    special_tokens_list: list = field(default_factory=list)
    output_dir: str = "outputs/"
    overwrite_output_dir: bool = True
    model_name: str = None
    tokenizer_name: str = None
    reprocess_input_data: bool = False
    silent: bool = False
    no_cache: bool = False
    cache_dir: str = "cache_dir/"
    no_save: bool = False
    top_k: float = 40
    top_p: float = 0.9
    model_name_or_path: Optional[str] = field(
        default="shibing624/chinese-alpaca-plus-7b-hf"
    )
    use_peft: bool = True
    peft_type: str = "LORA"
    peft_bin_name: str = "adapter_model.bin"
    lora_r: int = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ["all"]  # ["all"] or ["k_proj"]
    lora_bias = "none"
    adalora_init_r: int = 12
    adalora_tinit: int = 200
    adalora_tfinal: int = 1000
    adalora_delta_t: int = 10
    lora_beta: float = 0.85
    num_virtual_tokens: int = 20
    prompt_encoder_hidden_size: int = 128
    num_train_epochs = 3
    max_steps = -1
    per_device_train_batch_size = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps = 1
    save_total_limit = 10
    remove_unused_columns = False
    logging_steps = 50
    resume_from_checkpoint: str = None
    gradient_checkpointing: bool = True
    torch_compile: bool = False
    trust_remote_code: bool = True
    qlora: bool = False
    preprocessing_num_workers: int = 4
    prompt_template_name: str = "qwen"

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))


class Qwen2Model:

    def __init__(
        self,
        model_name: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct",
        peft_name: Optional[str] = None,
        args: Optional[dict] = None,
        use_cuda: Optional[bool] = True,
        cuda_device: Optional[int] = -1,
        **kwargs,
    ) -> None:
        self.args = ModelArgs()
        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ModelArgs):
            self.args = args

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if torch.cuda.is_available() > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        self.device_map = "auto"
        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
                    self.device_map = {"": int(cuda_device)}
            else:
                raise ValueError("CUDA is not available on this machine.")

        else:
            self.device = torch.device("cpu")
            self.device_map = {"": "cpu"}

        logger.info(f"Using device:{self.device}")
        if not use_cuda:
            self.args.fp16 = False  # disable fp16
            self.args.int8 = False  # disable int8
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info(f"world_size:{self.world_size}")
        self.ddp = self.world_size != 1
        if self.ddp:
            self.device_map = {"": self.local_rank}
        self.results = {}
        if model_name:
            self.args.model_name = model_name
        if self.args.bf16:
            self.args.fp16 = False
        if self.args.fp16:
            self.args.bf16 = False
        self.torch_dtype = (
            torch.bfloat16
            if self.args.bf16
            else (torch.float16 if self.args.fp16 else torch.float32)
        )
        self.config = Qwen2Config.from_pretrained(
            model_name,
            trust_remote_code=self.args.trust_remote_code,
            torch_dtype=self.torch_dtype,
            **kwargs,
        )
        self.model = Qwen2ForCausalLM.from_pretrained(
            model_name,
            config=self.config,
            load_in_8bit=self.args.int8,
            load_in_4bit=self.args.int4,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            device_map=self.device_map,
            trust_remote_code=self.args.trust_remote_code,
            quantization_config=(
                BitsAndBytesConfig(
                    load_in_4bit=self.args.int4,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.torch_dtype,
                )
                if self.args.qlora
                else None
            ),
        )
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            model_name, trust_remote_code=self.args.trust_remote_code
        )
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token = "</s>"
            logger.debug(f"Add eos token: {self.tokenizer.eos_token}")
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.unk_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.debug(f"Add pad token:{self.tokenizer.pad_token}")
        if self.model.config.architectures[0] == "Qwen2ForCasualLM":
            self.tokenizer.padding_side = "left"
        self.peft_name = peft_name
        if self.args.use_peft and self.peft_name:
            self.load_peft_model()

    def load_peft_model(self):
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(
            self.model,
            self.peft_name,
            torch_dtyte=self.torch_dtype,
            device_map=self.device_map,
        )
        self.model = self.model.merge_and_unload()
        logger.info(f"Loaded peft model from {self.peft_name}")

    def find_all_linear_names(self, int4=False, int8=False):
        cls = torch.nn.Linear
        if int4 or int8:
            import bitsandbytes as bnb

            if int4:
                cls = bnb.nn.Linear4bit
            elif int8:
                cls = bnb.nn.Linear8bitLt
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, cls):
                if "lm_head" in name:
                    continue
                if "output_layer" in name:
                    continue
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        return sorted(lora_module_names)

    def train_model(
        self,
        train_data,
        output_dir=None,
        args=None,
        eval_data=None,
        verbose=True,
        **kwargs,
    ):
        from peft import (
            get_peft_model,
            LoraConfig,
            TaskType,
            PeftModel,
            prepare_model_for_kbit_training,
            set_peft_model_state_dict,
        )

        if args:
            self.args.update_from_dict(args)
        if eval_data is None:
            logger.debug(
                "eval data is not specified, Pass eval data to model.train_model() if using evaluate."
            )
        if not output_dir:
            output_dir = self.args.output_dir
        if (
            os.path.exists(output_dir)
            and os.listdir(output_dir)
            and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
            )
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            logging_dir=f"{output_dir}/logs",
            logging_steps=self.args.logging_steps,
            max_steps=self.args.max_steps,
            per_device_eval_batch_size=self.args.per_device_train_batch_size,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_checkpointing=self.args.gradient_checkpointing,
            torch_compile=self.args.torch_compile,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_steps=self.args.warmup_steps,
            save_steps=self.args.save_steps,
            optim=self.args.optimizer,
            save_strategy=self.args.save_strategy,
            eval_strategy="steps" if eval_data is not None else "no",
            eval_steps=self.args.eval_steps if eval_data is not None else None,
            load_best_model_at_end=True if eval_data is not None else False,
            ddp_find_unused_parameters=False if self.ddp else None,
            save_total_limit=self.args.save_total_limit,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            remove_unused_columns=self.args.remove_unused_columns,
            report_to=self.args.report_to,
            overwrite_output_dir=self.args.overwrite_output_dir,
            no_cuda=True if self.device == "cpu" else False,
            **kwargs,
        )
        resume_from_checkpoint = self.args.resume_from_checkpoint
        if self.args.qlora and (
            len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()
        ):
            logger.warning("FSDP and ZeRO3 are both currently incompatible with QLoRA.")
        if "all" in self.args.lora_target_modules:
            self.args.lora_target_modules = self.find_all_linear_names(
                self.args.int4, self.args.int8
            )
        if self.args.use_peft:
            if self.args.int8 or self.args.int4:
                self.model = prepare_model_for_kbit_training(
                    self.model, self.args.gradient_checkpointing
                )
            peft_type = self.args.peft_type.upper()
            logger.info(f"Using PEFT type:{peft_type}")
            if peft_type == "LORA":
                logger.debug(
                    f"Using list modules for LoRA: {self.args.lora_target_modules}"
                )
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=self.args.lora_r,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=self.args.lora_target_modules,
                    bias=self.args.lora_bias,
                )
            elif peft_type == "ADALORA":
                from peft import AdaLoraConfig

                logger.debug(
                    f"Using list modules for AdaLoRA: {self.args.lora_target_modules}"
                )
                peft_config = AdaLoraConfig(
                    init_r=self.args.adalora_init_r,
                    r=self.args.adalora_init_r,
                    beta1=self.args.lora_beta,
                    beta2=self.args.lora_beta,
                    tinit=self.args.adalora_tinit,
                    tfinal=self.args.adalora_tfinal,
                    deltaT=self.args.adalora_delta_t,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=self.args.lora_target_modules,
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                )
            elif peft_type == "PROMPT_TUNING":
                from peft import PromptTuningConfig

                peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=self.args.num_virtual_tokens,
                )
            elif peft_type == "P_TUNING":
                from peft import PromptEncoderConfig

                peft_config = PromptEncoderConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=self.args.num_virtual_tokens,
                    encoder_hidden_size=self.args.prompt_encoder_hidden_size,
                )
            elif peft_type == "PREFIX_TUNING":
                from peft import PrefixTuningConfig

                peft_config = PrefixTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=self.args.num_virtual_tokens,
                    encoder_hidden_size=self.args.prompt_encoder_hidden_size,
                    prefix_projection=True,
                )
                self.model.gradient_checkpointing_disable()
            else:
                logger.warning(f"Wrong type of peft. Set to default lora")
                logger.debug(
                    f"Using list modules for LoRA: {self.args.lora_target_modules}"
                )
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=self.args.lora_r,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=self.args.lora_target_modules,
                    bias=self.args.lora_bias,
                )
            if isinstance(self.model, PeftModel):
                logger.debug("Merge peft weights to base model")
                self.model = self.model.merge_and_unload()
            self.model = get_peft_model(self.model, peft_config)

            for param in filter(lambda p: p.requires_grad, self.model.parameters()):
                param.data = param.data.to(torch.float32)

            if resume_from_checkpoint:
                checkpint_name = os.path.join(
                    resume_from_checkpoint, "pytorch_model.bin"
                )
                if not os.path.exists(checkpint_name):
                    checkpint_name = os.path.join(
                        resume_from_checkpoint, "adapter_model.bin"
                    )
                    resume_from_checkpoint = False
                if os.path.exists(checkpint_name):
                    logger.info(f"Restarting from {checkpint_name}")
                    adapter_weights = torch.load(checkpint_name, map_location="cpu")
                    set_peft_model_state_dict(self.model, adapter_weights)
                else:
                    logger.warning(f"Checkpoint {checkpint_name} not found")
                    resume_from_checkpoint = None
            self.model.print_trainable_parameters()
        else:
            logger.info("Fune-tuning method: Full parameters training")
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Tokenizer: {self.tokenizer}")

        train_dataset = self.load_and_cache_examples(train_data)
        if verbose:
            logger.debug(
                f"train_dataset len: {len(train_dataset)},train_dataset[0]: {train_dataset[0]}]"
            )
            logger.debug("Tokenized training example:")
            logger.debug(
                f"Decode input_ids[0]:{self.tokenizer.decode(train_dataset[0]['input_ids'])}"
            )
            replaced_labels = [
                label if label != IGNORE_INDEX else self.tokenizer.pad_token_id
                for label in list(train_dataset[0]["labels"])
            ]
            logger.debug(f"Decode labels[0]: {self.tokenizer.decode(replaced_labels)}")
        eval_dataset = None
        if eval_data is not None:
            eval_dataset = self.load_and_cache_examples(eval_data, evaluate=True)
            if verbose:
                logger.debug(
                    f"eval_dataset len: {len(eval_dataset)}, eval_dataset[0]: {eval_dataset[0]}"
                )
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        if training_args.local_rank <= 0:
            logger.info(f"Training/evaluation parameters {training_args}")
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        else:
            self.model.config.use_cache = True
        self.model.enable_input_require_grads()
        if not self.ddp and torch.cuda.device_count() > 1:
            # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            self.model.is_parallelizable = True
            self.model.model_parallel = True

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, label_pad_token_id=IGNORE_INDEX
        )
        trainer = SavePeftModelTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if eval_data is not None else None,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        # Training
        logger.info("*** Train ***")
        sample = next(iter(trainer.get_train_dataloader()))
        logger.debug(f"Train dataloader example: {sample}")
        logger.debug(
            f"Detail input_ids: {sample['input_ids'][:3]}, \nlabels: {sample['labels'][:3]}"
        )
        logger.debug(
            f"Decode input_ids[0]: {self.tokenizer.decode(sample['input_ids'][0])}"
        )
        replaced_labels = [
            label if label != IGNORE_INDEX else self.tokenizer.pad_token_id
            for label in sample["labels"][0]
        ]
        logger.debug(f"Decode labels[0]: {self.tokenizer.decode(replaced_labels)}")

        (global_step, training_loss, metrics) = trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )
        self.results.update(metrics)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        self.model.config.use_cache = True
        trainer.save_state()
        self.save_model(model=self.model)

        if eval_data is not None:
            logger.info("*** Evaluate ***")
            if self.args.fp16:
                self.model.half()

            metrics = trainer.evaluate(metric_key_prefix="eval")
            metrics["eval_samples"] = len(eval_dataset)
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity
            logger.debug(f"eval metrics: {metrics}")
            self.results.update(metrics)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        if verbose and training_args.local_rank <= 0:
            logger.debug(f"metrics: {self.results}")
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    self.args.model_name, output_dir
                )
            )
        return global_step, training_loss

    @torch.inference_mode()
    def predict(
        self,
        sentences: List[str],
        skip_prompt: bool = True,
        prompt_template_name: str = None,
        max_length: int = None,
        do_sample: bool = None,
        temperature: float = None,
        repetition_penalty: float = None,
        eval_batch_size: int = None,
        system_prompt: str = None,
        **kwargs,
    ) -> List[str]:
        """
        Performs predictions on a list of text.

        Args:
            sentences: A python list of text (str) to be sent to the model for prediction. Note that the prefix should be prepended to the text.
            skip_prompt: Whether to skip the prompt when generating text.
            prompt_template_name: The name of the prompt template to use.
            max_length: The maximum length of the generated text.
            do_sample: Whether or not to use sampling ; use greedy decoding otherwise.
            temperature: The value used to module the next token probabilities.
            repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty.
            eval_batch_size: Batch size to use for evaluation.
            system_prompt: The system prompt to use for generation.
            **kwargs: Additional arguments for generating sequences.

        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        self.model.eval()
        if self.args.fp16:
            self.model.half()
        prompt_template = get_conv_template(
            prompt_template_name or self.args.prompt_template_name
        )
        if not eval_batch_size:
            eval_batch_size = self.args.eval_batch_size

        all_outputs = []
        # Batching
        for batch in tqdm(
            [
                sentences[i : i + eval_batch_size]
                for i in range(0, len(sentences), eval_batch_size)
            ],
            desc="Generating outputs",
            disable=self.args.silent,
        ):
            generation_kwargs = dict(
                max_new_tokens=(
                    max_length if max_length is not None else self.args.max_length
                ),
                do_sample=do_sample if do_sample is not None else self.args.do_sample,
                temperature=(
                    temperature if temperature is not None else self.args.temperature
                ),
                repetition_penalty=(
                    repetition_penalty
                    if repetition_penalty is not None
                    else self.args.repetition_penalty
                ),
            )

            if prompt_template_name:
                prompts = [
                    prompt_template.get_prompt(
                        messages=[[s, ""]], system_prompt=system_prompt
                    )
                    for s in batch
                ]
                inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)
                outputs = self.model.generate(input_ids, **generation_kwargs, **kwargs)
            else:
                conversation = []
                for s in batch:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": s})
                    conversation.append(messages)
                inputs = self.tokenizer.apply_chat_template(
                    conversation=conversation,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                )
                input_ids = inputs.to(self.device)
                outputs = self.model.generate(input_ids, **generation_kwargs, **kwargs)

            for input_text, generated_sequence in zip(batch, outputs):
                # Decode text
                prompt_len = len(input_ids[0])
                generated_sequence = generated_sequence[prompt_len:]
                gen_text = self.tokenizer.decode(
                    generated_sequence, skip_special_tokens=True
                )
                gen_text = gen_text.strip()
                # logger.debug(f"input_text: {input_text}, gen_text: {gen_text}")
                all_outputs.append(gen_text)

        return all_outputs

    def save_model(
        self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None
    ):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if model and not self.args.no_save:
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

    def load_and_cache_examples(
        self, data, evaluate=False, no_cache=False, verbose=True, silent=False
    ):
        tokenizer = self.tokenizer
        args = self.args
        if not no_cache:
            no_cache = args.no_cache
        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)
        mode = "dev" if evaluate else "train"
        if args.dataset_class:
            CustomDataset = args.dataset_class
            return CustomDataset(tokenizer, args, data, mode)
        else:
            return SupervisedDataset(tokenizer, args, data, mode)


class SavePeftModelTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)
