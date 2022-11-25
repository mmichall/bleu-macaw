import argparse
import logging
import random

import torch
import os
from typing import Callable
import numpy as np
import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, TextGenerationPipeline, GPT2Config, \
    GPT2LMHeadModel, GPT2Tokenizer
from transformers.pipelines.text_generation import ReturnType

from modified_gpt2 import GPT2ParaphrasingLM

from datasets import load_dataset
from config import results_dir_path
from util import filter_best

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
        self.sentnce_sim_encoder = SentenceTransformer(args.sentnce_sim_encoder)

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

    def __init__(self, model_path: str, encoder_name: str, sim_encoder_name: str):
        self.model_path = model_path
        self.encoder_name = encoder_name
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.encoder = SentenceTransformer(encoder_name) if encoder_name else None
        self.encoder.to('cuda')
        self.sim_encoder = SentenceTransformer(sim_encoder_name) if sim_encoder_name else None
        self.generator = ParaphrasingPipeline(model=self.model, tokenizer=self.tokenizer, sentnce_sim_encoder=config.)

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        return tokenizer

    def _load_model(self):
        # model = GPT2LMHeadModel.from_pretrained(self.model_path) # baseline
        model = GPT2ParaphrasingLM.from_pretrained(self.model_path)  # gpt2 paraphrasing gpt2-para
        model.eval()
        return model

    def generate(self, sentence: str, strategy: str = "beam_search", _filter_best: bool = False, k: int = 10):
        assert strategy in ("sampling", "beam_search")
        if self.encoder:
            embeddings = self.encoder.encode([sentence], convert_to_tensor=True)
        else:
            embeddings = None
        strategy: Callable = self._sampling if strategy == "sampling" else self._beam_search
        outputs = strategy(sentence, 1, 128, k, embeddings)
        outputs_sent = {sent.get("generated_text") for sent in outputs if sent != sentence}
        # outputs_sent = [sent.split(' paraphrased: ')[1] for sent in outputs_sent]
        if len(outputs_sent) == 0: return [sentence], sentence
        return outputs_sent, filter_best(sentence, outputs_sent, ...) if _filter_best else ''

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

    def _beam_search(self, embeddings, num_beams, num_return_sequences):
        return self.generator("",
                              max_length=64,
                              num_beams=num_beams,
                              num_return_sequences=num_return_sequences,
                              temperature=1.0,
                              num_beam_groups=5,
                              repetition_penalty=1.4,
                              diversity_penalty=0.4,
                              no_repeat_ngram_size=2,
                              early_stopping=True,
                              length_penalty=0.0,
                              sentence_embedding=embeddings
                              )


def _process_text(sent: str):
    sent = sent.replace("<br />", " ")
    # paraphrased: <quora>
    # text = ''.join(['<quora>', text, '<|endoftext|>']) # quora -para
    # text = ''.join(['<quora>', text, ' paraphrased: ']) # quora -para
    # text = ' '.join([text, '<|endoftext|>'])
    if sent.endswith('!'):
        return '<|exclamation|> ' + sent + ' <|endoftext|>'
    if sent.endswith('?'):
        return '<|question|> ' + sent + ' <|endoftext|>'
    else:
        return '<|startoftext|> ' + sent + ' <|endoftext|>'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_file",
        type=str, required=False, help="The input evaluation data file (a text file)."
    )
    parser.add_argument(
        "--hg_dataset_name",
        type=str, required=False, help="Huggingface dataset name"
    )
    parser.add_argument(
        "--text_column_name",
        type=str, required=False, help="Text column name"
    )
    parser.add_argument(
        "--output_dir",
        default='./output_dir',
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--model_encoder_name",
        default="",
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--model_sim_encoder_name",
        required=True,
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--cache_dir",
        default="./.cache",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()


    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if not args.data_file and not args.hg_dataset_name:
        raise ValueError(
            "Neither data_file or hg_dataset_name specified.".format(
                args.output_dir
            )
        )

    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Process, device: %s, n_gpu: %s, 16-bits training: %s",
        device,
        1,
        args.fp16,
    )
    set_seed(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.hg_dataset_name:
        data_tag = args.hg_dataset_name.split('/')[1]
        dataset = load_dataset("HHousen/msrp")
        # msrp = load_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT")
        # msrp = load_dataset("cestwc/adapted-paranmt5m")
        dataset = dataset["train"]
        lines = []
        for item in dataset:
            line = item[args.text_column_name] + '\t' + '[###]'
            lines.append(line)

    if args.data_file:
        data_tag = args.data_file.split('.')[0]
        with open(args.data_file, "r", encoding='utf-8') as f:
            lines = [line.rstrip() for line in f]
            if '\t' not in lines[0]:
                raise Exception('Wrong data format! Test file text format is like: input\\t[ref0,...refN]')


    logger.info("Evaluation parameters %s", args)

    generator = ParaphrasingGenerator(args.model_name_or_path, args.model_encoder_name, args.model_sim_encoder_name)

    with open(f'{results_dir_path}/{args.model_encoder_name}-{data_tag}.tsv', "w", encoding='utf-8') as f:
        for line in tqdm.tqdm(lines):
            input, refs = line.split('\t')
            orginal_input = input

            input = _process_text(input)
            cands, the_best = generator.generate(input, strategy="beam_search", filter_best=True, k=5)
            cands = [cand.replace('\n', ' ') for cand in cands]
            the_best = the_best.replace('\n', ' ')
            print(input, the_best)
            print('\t'.join([orginal_input, the_best, str(cands), str(refs)]), file=f)
