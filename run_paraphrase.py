import argparse
import logging
import random

import datasets
import torch
import os
from typing import Callable
import numpy as np
import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, TextGenerationPipeline, GPT2Config, \
    GPT2LMHeadModel, GPT2Tokenizer, RobertaTokenizer, RobertaModel, RobertaConfig, AutoModel, \
    T5ForConditionalGeneration, Text2TextGenerationPipeline
from transformers.pipelines.text_generation import ReturnType

from modified_gpt2 import GPT2ParaphrasingLM

from config import results_dir_path
from util import filter_best

from config import data_path

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

chars_to_strip = {'"', '``', '”', '\'\'', '\'', '[…]', '[serious]', '[Serious]', '-', '_', '|', '#', '<', '>', '“',
                  '\"', ','}
chars_to_remove = {'\n', '\t', '....', '...', '..', '~', '•', '·', '   ', '  '}
MAX_SENTS = 10


def preprocess(text):
    # text = text.strip()
    # for char in chars_to_strip: text = text.strip(char)
    for char in chars_to_remove: text = text.replace(char, ' ')
    text.replace('!!!', '!').replace('!!', '!')
    text.replace('???', '?').replace('??', '?')
    text.replace('?.', '?').replace('.?', '.')
    text.replace('!.', '!').replace('.!', '.')
    return ' '.join(text.strip().split())


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
            truncation=True,
            add_special_tokens=True,
            return_tensors=self.framework,
            max_length=256,
            # truncation=True
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

    def __init__(self, model_path: str, encoder_name):
        self.model_path = model_path
        self.encoder_name = encoder_name
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self.encoder = SentenceTransformer(encoder_name) if encoder_name else None
        self.encoder.to('cuda')
        self.generator = ParaphrasingPipeline(model=self.model, tokenizer=self.tokenizer, device='cuda')

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        tokenizer.model_max_length = 256
        return tokenizer

    def _load_model(self):
        # model = GPT2LMHeadModel.from_pretrained(self.model_path) # baseline
        model = GPT2ParaphrasingLM.from_pretrained(self.model_path)  # .to('cuda')  # gpt2 paraphrasing gpt2-para
        # T5
        # model = T5ForConditionalGeneration.from_pretrained(self.model_path)
        model.eval()
        return model

    # def mean_pooling(self, model_output, attention_mask):
    #     token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    #     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate(self, sentence: str, strategy: str = "beam_search", _filter_best: bool = False, k: int = 10):
        assert strategy in ("sampling", "beam_search")
        if self.encoder:
            embeddings = self.encoder.encode([sentence], convert_to_tensor=True)  # .to('cuda')
        else:
            embeddings = None

        # config1 = RobertaConfig.from_pretrained("roberta-base")
        # config1.output_hidden_states = True
        # self.xlm_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        # self.xlm_model = RobertaModel.from_pretrained("roberta-base", config=config1).to('cuda')
        # encoded_input = self.xlm_tokenizer([sentence], truncation=True, padding=True, return_tensors='pt').to('cuda')
        # forward pass
        # xlm_embedding = self.xlm_model.embeddings(encoded_input)
        # embeddings = np.add(embeddings, _noise)
        # with torch.no_grad():
        #     model_output = self.xlm_model(**encoded_input)
        # xlm_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # noising
        # embeddings = embeddings + torch.multiply(embeddings, 0.1*torch.randn(768).cuda())
        strategy: Callable = self._sampling if strategy == "sampling" else self._beam_search
        with torch.no_grad():
            outputs = strategy(sentence, embeddings)
        outputs_sent = [sent.get("generated_text") for sent in outputs if sent != sentence]
        # outputs_sent = [sent.split(' paraphrased: ')[1] for sent in outputs_sent]
        if len(outputs_sent) == 0: return [sentence], sentence
        return outputs_sent, filter_best(sentence, list(outputs_sent)) if _filter_best else '###'

    def _sampling(self, sentence, embeddings):
        return self.generator("<|endoftext|>",
                              do_sample=True,
                              repetition_penalty=1.8,  # The parameter for repetition penalty. 1.0 means no penalty.
                              add_special_tokens=True,
                              num_return_sequences=1,
                              sentence_embedding=embeddings,
                              temperature=0.8,
                              top_p=0.75,
                              # sentence_embedding=embeddings  # gpt2-para
                              )

    def _beam_search(self, sentence, embeddings):
        return self.generator('',
                              num_beams=5,
                              num_return_sequences=5,
                              num_beam_groups=5,
                              diversity_penalty=0.6,
                              no_repeat_ngram_size=2,
                              length_penalty=0.0,
                              # early_stopping="never",
                              # length_penalty=1.2, # length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences
                              sentence_embedding=embeddings,
                              # renormalize_logits=True,
                              max_length=256,
                              # forced_eos_token_id=50256
                              # max_new_tokens=64
                              )


def _preprocess_text(sent: str):
    sent = preprocess(sent)
    # sent = sent.replace("<br />", " ")
    # sent = ('question: ' if sent.endswith('?') else '<|endoftext|>') + sent
    return sent
    # if sent.endswith('!'):
    #     return '<|exclamation|> ' + sent + ' <|endoftext|>'
    # if sent.endswith('?'):
    #     return '<|question|> ' + sent + ' <|endoftext|>'
    # else:
    #     return '<|startoftext|> ' + sent + ' <|endoftext|>'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_dataset_name",
        # default='ChristophSchuhmann/MS_COCO_2017_URL_TEXT',
        default='cnn_news',
        type=str, required=False, help="Huggingface dataset name"
    )
    parser.add_argument(
        "--data_file",
        type=str, required=False, help="The input evaluation data file (a text file)."
    )
    parser.add_argument(
        "--model_name_or_path",
        required=False,
        # default='gpt2-medium',
        # default="E:\\output",
        default=r"E:\.cache\bleu-macaw-pretrained-cnn_news\gpt-medium\quora-paraphrase-mpnet-base-v2-cnn_news",
        help="The model checkpoint for weights initialization.",
    )
    # paraphrase-xlm-r-multilingual-v1
    parser.add_argument(
        "--model_encoder_name",
        required=False,
        type=str,
        # default='all-roberta-large-v1',
        # default='paraphrase-xlm-r-multilingual-v1',
        # default='thenlper/gte-base',
        default='paraphrase-mpnet-base-v2',
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument("--seed", type=int, default=42, help="randomseed for initialization")

    args = parser.parse_args()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args)

    data_tag = ''
    if args.hf_dataset_name:
        if not os.path.exists(f'{data_path}/{args.hf_dataset_name}'):
            raise Exception(
                f'{data_path}/{args.hf_dataset_name} doesn\'t exist. Please generate data files with ./dataset/generate.pl --dataset_name {args.hf_dataset_name}')
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

    generator = ParaphrasingGenerator(args.model_name_or_path, args.model_encoder_name)

    with open(f'{results_dir_path}/{generator.encoder_name}-{data_tag}_cnn_news-base.tsv', "w", encoding='utf-8') as f:
        for line in tqdm.tqdm(lines):
            input, refs = line.split('\t')
            orginal_input = input
            input = _preprocess_text(input)
            cands, the_best = generator.generate(input, strategy="beam_search", _filter_best=True, k=5)
            cands = [cand.replace('\n', ' ') for cand in cands]
            the_best = the_best.replace('\n', ' ')
            print(input)
            for i, cand in enumerate(cands):
                print(str(i) + '. ' + cand)
            print('-------------------------')
            print('\t'.join([orginal_input, the_best, str(cands), str(refs)]), file=f)
