#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import argparse
import logging
import math
import os
import sys

from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset, load_from_disk, load_metric

import transformers
from transformers import (
    CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig, AutoTokenizer,
    HfArgumentParser, Trainer, PreTrainedTokenizer, GPT2Config, Seq2SeqTrainingArguments,
    set_seed
)
from transformers.file_utils import PaddingStrategy
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
from modified_gpt2 import SentenceTransformerTokenizerWrapper, GPT2ParaphrasingLM, \
    ParaphrasingDataCollator

import nltk

nltk.download('punkt')

logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_DATASETS_CACHE"] = "./cache"
os.environ["TRANSFORMERS_CACHE"] = "./cache"

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
                    "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=128,
        metadata={
            "help": "Optional input sequence length after tokenization. "
                    "The training dataset will be truncated in block of this size for training. "
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=False, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main(_args):


    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    sentence_encoder = _args.sentence_transformer
    # corpus_name = _args.corpus_name
    tag = _args.tag
    gpt2_model = _args.gpt2_model
    corpus_limit = _args.corpus_limit
    output_dir = _args.output_dir

    corpus_cache_name = f'quora-{sentence_encoder}' + (
        f'-{str(corpus_limit / 1_000_000)}M' if corpus_limit else '') + f'-{tag}'

    lang = "english"

    args = [
        "--do_train",
        "--do_eval",
        f"--output_dir={output_dir}/gpt-medium/{corpus_cache_name}/",
        f"--logging_dir='./gpt-medium/{corpus_cache_name}/'",
        "--per_device_train_batch_size=32",
        "--per_device_eval_batch_size=8",
        "--gradient_accumulation_steps=1",
        # "--fp16",
        "--num_train_epochs=8",
        # "--num_train_epochs=5",
        "--max_steps=180_000",
        "--save_steps=10_000",
        "--eval_steps=10_000",
        "--warmup_steps=0",
        "--weight_decay=0.01", # 0.01
        "--evaluation_strategy=steps",
        "--overwrite_output_dir",
        "--model_type=gpt2",
        "--learning_rate=5e-6",
        f"--model_name_or_path={gpt2_model}",
        # "--torch_compile=True",  # optimizations
        '--optim=adamw_hf'  # improved optimizer
    ]
    if lang == "english":
        # args.append(f"--dataset_name={corpus_name}")
        args.append(f"--dataset_name=multi")
        args.append("--dataset_config_name=plain_text")
    elif lang == "polish":
        args.append("--model_name_or_path=dkleczek/papuGaPT2")
        args.append("--train_file=polish.txt")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        # raw_datasets = load_dataset('openwebtext', streaming=True)
        # raw_datasets = load_dataset('oscar', 'unshuffled_deduplicated_en', streaming=True)
        # from corpus_from_file import multicorpus

        # data_files = {"train": "C:/Projekty/vae-text-generator/.data/ChristophSchuhmann/MS_COCO_2017_URL_TEXT/unsupervised.txt",
        #                "valid": "C:/Projekty/vae-text-generator/.data/ChristophSchuhmann/MS_COCO_2017_URL_TEXT/dev_unsupervised.txt"}
        # multicorpus = load_dataset('text', data_files=data_files, streaming=True).shuffle(buffer_size=1_000)
        # raw_datasets_valid = multicorpus['valid']

        from corpus_from_file import multicorpus # cnn_daily_mail_corpus
        raw_datasets = multicorpus['train']
        raw_datasets_valid = multicorpus['valid']

        # valid_no = int(50_000)

        # raw_datasets_valid = raw_datasets.take(valid_no)
        # raw_datasets["train"] = raw_datasets["train"].skip(valid_no)

        # raw_datasets = raw_datasets.with_format("torch")
        # column_names = raw_datasets["train"].column_names
        # column_names = ['id', 'text']
        # text_column_name = "text" if "text" in column_names else column_names[0]

        # if _args.debug or not os.path.exists(f'{output_dir}/.cache/{corpus_cache_name}'):
        # splitter = DatasetSentenceSplitter(raw_datasets, text_column_name)
        # raw_datasets = splitter.split()
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config: GPT2Config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    # config.scale_attn_by_inverse_layer_idx = True
    # config.reorder_and_upcast_attn = True

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None
    }
    if model_args.tokenizer_name:
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                                       **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    tokenizer.pad_token = tokenizer.eos_token

    if model_args.model_name_or_path:
        model = GPT2ParaphrasingLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        ).cuda()

        # modules_to_freeze = [model.transformer.h[i] for i in range(0, len(model.transformer.h))]
        # modules_to_freeze = []
        # modules_to_freeze.extend([model.transformer.wte])  # model.lm_head, model.transformer.wpe
        # for module in modules_to_freeze:
        #    for param in module.parameters():
        #        param.requires_grad = False  # Actual freezing operation

    else:
        model = GPT2ParaphrasingLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params")

    if (not _args.debug) and os.path.exists(f'{output_dir}/.cache/{corpus_cache_name}'):
        tokenized_datasets = load_from_disk(f'{output_dir}/.cache/{corpus_cache_name}', )
        print(f'Corpus load from {output_dir}/.cache/{corpus_cache_name}')
    else:
        print(f'Preparing corpus {output_dir}/.cache/{corpus_cache_name}')
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        # raw_datasets = raw_datasets.with_format("torch")
        # if training_args.do_train:
        #     column_names = ['id', 'text']
        # else:
        #     column_names = ['id', 'text']
        text_column_name = "text"  # if "text" in column_names else column_names[0]
        paraphrases_column_name = "paraphrases"
        # text_column_name = "text"

        # splitter = DatasetSentenceSplitter(raw_datasets, text_column_name)
        # raw_datasets = splitter.split()

        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_ffunction
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        tokenizer_wrapper = SentenceTransformerTokenizerWrapper(tokenizer, text_column_name, paraphrases_column_name, sentence_encoder)

        # def _process_text(text: str):
        #     return text.replace("<br />", ". ").replace("<br/>", ". ").replace("?.", "? ") \
        #         .replace("\\n\\n"," ").replace("\\n", " ").replace("\\t", " ").replace(".?", ".")

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer_wrapper.tokenize(examples)
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return output

        # if "validation" not in raw_datasets.keys():
        #     corpus_no = len(raw_datasets)
        #     valid_no = int(len(raw_datasets) * 5 / 100)
        #     raw_datasets["train"] = raw_datasets["train"].select(range(valid_no, corpus_no + valid_no))
        #     raw_datasets["validation"] = raw_datasets["train"].select(range(0, valid_no))
        # if "unsupervised" in raw_datasets.keys():
        #     raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], raw_datasets["unsupervised"]])
        #     del raw_datasets["unsupervised"]

        with training_args.main_process_first(desc="dataset map tokenization"):

            # raw_datasets = raw_datasets.map(
            #     get_sentence_function,
            #     batched=True,
            #     # num_proc=data_args.preprocessing_num_workers,
            #     # load_from_cache_file=not data_args.overwrite_cache,
            #     # desc="Running sentenciger on datset",
            # )

            print('Running tokenizer on dataset')
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                batch_size=1000,
                load_from_cache_file=False,
                # load_from_cache_file=True,
                # remove_columns=column_names
                # remove_columns=['id', 'prompt', 'story'],
                # num_proc=data_args.preprocessing_num_workers,
                # desc="Running tokenizer on dataset"
            )

            tokenized_datasets_valid = raw_datasets_valid.map(
                tokenize_function,
                batched=True,
                batch_size=1000,
                load_from_cache_file=False,
                # remove_columns=column_names
                # num_proc=data_args.preprocessing_num_workers,
                # load_from_cache_file=not data_args.overwrite_cache,
                # desc="Running tokenizer on dataset"
            )
        # if data_args.block_size is None:
        #     block_size = tokenizer.model_max_length
        #     if block_size > 32:
        #         logger.warning(
        #             f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
        #             "Picking 1024 instead. You can change that default value by passing --block_size xxx."
        #         )
        #         block_size = 32
        # else:
        #     if data_args.block_size > tokenizer.model_max_length:
        #         logger.warning(
        #             f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
        #             f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        #         )
        #     block_size = min(data_args.block_size, tokenizer.model_max_length)

        # if "validation" not in raw_datasets.keys():
        #     raw_datasets["train"] = raw_datasets["train"].select(range(10000, corpus_limit))
        #     raw_datasets["validation"] = raw_datasets["train"].select(range(0, 10000))
        # if "unsupervised" in raw_datasets.keys():
        #     raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], raw_datasets["unsupervised"]])
        #     del raw_datasets["unsupervised"]

        # if not _args.debug:
        #     tokenized_datasets.save_to_disk(f'{output_dir}/.cache/{corpus_cache_name}')
        #     print(f'Corpu saved in {output_dir}/.cache/{corpus_cache_name}')

    lm_datasets = tokenized_datasets.with_format("torch")
    lm_datasets_valid = tokenized_datasets_valid.with_format("torch")

    if training_args.do_train:
        # if "train" not in tokenized_datasets:
        #     raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets  # ["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # train_dataset = train_dataset.map(lambda s: s, batched=True, remove_columns=['label'])

    if training_args.do_eval:
        # if "validation" not in tokenized_datasets:
        #     raise ValueError("--do_eval requires a validation dataset")
        # eval_dataset = lm_datasets["validation"]
        eval_dataset = lm_datasets_valid
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        # eval_dataset = eval_dataset.map(lambda s: s, batched=True, remove_columns=['label'])

    # training_args.evaluation_strategy = IntervalStrategy.STEPS
    # training_args.eval_steps = 1000

    # metric = evaluate.load("bleu")
    def compute_metrics(eval_preds):
        # preds, labels = eval_preds
        # if isinstance(preds, tuple):
        #     preds = preds[0]
        # print('preds:', preds)
        # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        #
        # # Some simple post-processing
        # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        #
        # result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        # result = {"bleu": result["score"]}
        #
        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        # result = {k: round(v, 4) for k, v in result.items()}
        # return result
        return {'f1': 1}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=ParaphrasingDataCollator(tokenizer, padding=PaddingStrategy.LONGEST, pad_to_multiple_of=8)
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        print("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        print("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        print("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--gpt2_model',
                        default=r'gpt2', type=str,
                        help='HuggingFace GPT2 model version name.')
    parser.add_argument('--sentence_transformer', required=False, type=str,
                        help='HuggingFace Sentence-Transformer model name.', default='paraphrase-mpnet-base-v2')
    # parser.add_argument('--corpus_name', default='openwebtext') #multi
    parser.add_argument('--corpus_limit', type=int, default=0)
    parser.add_argument('--output_dir', required=False, type=str, default='E://.cache/bleu-macaw-pretrained-cnn_news')
    parser.add_argument('--tag', default='cnn_news', help='Experiment\'s tag.')
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()

    main(args)
