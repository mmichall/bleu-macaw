import argparse
import logging
import random

import torch
import os
from typing import Callable
import numpy as np
import tqdm
from transformers import AutoTokenizer, TextGenerationPipeline, GPT2Config, \
    GPT2LMHeadModel, GPT2Tokenizer
from transformers.pipelines.text_generation import ReturnType

from config import results_dir_path

from config import data_path

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


class ParaphrasingPipeline(TextGenerationPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, prompt_text, prefix=""):
        inputs = self.tokenizer(
            prefix + prompt_text,
            padding=True,
            add_special_tokens=True,
            return_tensors=self.framework,
            max_length=512,
            truncation=True
        )
        inputs["prompt_text"] = prompt_text
        return inputs

    def _sanitize_parameters(
            self,
            return_full_text=None,
            return_tensors=None,
            return_text=None,
            return_type=None,
            clean_up_tokenization_spaces=None,
            prefix='',
            **generate_kwargs
    ):
        preprocess_params = {}
        if prefix is not None:
            preprocess_params["prefix"] = prefix
        if prefix:
            prefix_inputs = self.tokenizer(
                prefix, padding=False, return_tensors=self.framework
            )
            prefix_length = prefix_inputs["input_ids"].shape[-1]
            if "max_length" in generate_kwargs:
                generate_kwargs["max_length"] += prefix_length
            else:
                generate_kwargs["max_length"] = self.model.config.max_length + prefix_length

            if "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        forward_params = generate_kwargs

        postprocess_params = {}
        if return_full_text is not None and return_type is None:
            return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
        if return_tensors is not None and return_type is None:
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        return preprocess_params, forward_params, postprocess_params


class ParaphrasingGenerator:

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.generator = ParaphrasingPipeline(model=self.model, tokenizer=self.tokenizer)

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        return tokenizer

    def _load_model(self):
        model = GPT2LMHeadModel.from_pretrained(self.model_path) # baseline
        model.eval()
        return model

    def generate(self, sentence: str, strategy: str = "beam_search", _filter_best: bool = False, k: int = 10):
        assert strategy in ("sampling", "beam_search")
        strategy: Callable = self._sampling if strategy == "sampling" else self._beam_search
        outputs = strategy(sentence, num_beams=10, num_return_sequences=10)
        outputs_sent = [sent.get("generated_text") for sent in outputs if sent != sentence]
        if len(outputs_sent) == 0: return [sentence], sentence
        return outputs_sent, '###'

    def _sampling(self, sentence, k: int, embeddings):
        return self.generator(sentence if embeddings is None else '',
                              do_sample=True,
                              repetition_penalty=1.8,
                              add_special_tokens=True,
                              num_return_sequences=k,
                              temperature=0.8,
                              top_p=0.75,
                              sentence_embedding=embeddings  # gpt2-para
                              )

    def _beam_search(self, sentence, num_beams, num_return_sequences):
        return self.generator(sentence + ' >>>> ',
                              max_length=128,
                              num_beams=num_beams,
                              num_return_sequences=num_return_sequences,
                              temperature=1.0,
                              num_beam_groups=5,
                              repetition_penalty=1.2,
                              diversity_penalty=0.3,
                              no_repeat_ngram_size=2,
                              early_stopping=True,
                              length_penalty=0.0
                              )


def _preprocess_text(sent: str):
    return sent.replace("<br />", " ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_dataset_name",
        type=str, required=False, help="Huggingface dataset name"
    )
    parser.add_argument(
        "--data_file",
        type=str, required=False, help="The input evaluation data file (a text file)."
    )
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        type=str,
        help="The model checkpoint for weights initialization.",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args)

    if args.hf_dataset_name:
        data_tag = args.hf_dataset_name
        if not os.path.exists(f'{data_path}/{args.hf_dataset_name}'):
            raise Exception(f'{data_path}/{args.hf_dataset_name} doesn\'t exist. Please generate data files with ./dataset/generate.pl --dataset_name {args.hf_dataset_name}')
        else:
            args.data_file = f'{data_path}/{args.hf_dataset_name}'

    if args.data_file:
        data_tag = args.data_file.split('.')[0]
        # generate for test.txt file
        with open(f'{data_path}/{args.hf_dataset_name}/test.tsv', "r", encoding='utf-8') as f:
            lines = [line.rstrip() for line in f]
            if '\t' not in lines[0]:
                raise Exception('Wrong data format! Test file text format is like: input\\t[ref0,...refN]')

    logger.info("Evaluation parameters %s", args)

    generator = ParaphrasingGenerator(args.model_name_or_path)

    with open(f'{results_dir_path}/gpt2-base-{data_tag}.tsv', "w", encoding='utf-8') as f:
        for line in tqdm.tqdm(lines):
            input, refs = line.split('\t')
            orginal_input = input
            input = _preprocess_text(input)
            cands, the_best = generator.generate(input, strategy="beam_search", _filter_best=True, k=10)
            cands = [cand.replace('\n', ' ').split(' >>>> ')[1] for cand in cands]
            the_best = the_best.replace('\n', ' ')
            print(input, cands)
            print('\t'.join([orginal_input, the_best, str(cands), str(refs)]), file=f)
