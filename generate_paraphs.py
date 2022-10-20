import argparse
import logging
import pickle
import random
from ast import literal_eval

import torch
import os
from glob import glob
from typing import Set, Callable
import numpy as np
import tqdm
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from torch.utils.data import SequentialSampler, DataLoader
from transformers import AutoTokenizer, TextGenerationPipeline, Text2TextGenerationPipeline, GPT2Config, \
    OpenAIGPTConfig, BertConfig, RobertaConfig, DistilBertConfig, CamembertConfig, CamembertForMaskedLM, \
    GPT2LMHeadModel, GPT2Tokenizer, AutoModel, PreTrainedModel
from transformers.pipelines.text_generation import ReturnType

from modified_gpt2 import GPT2ParaphrasingLM

from datasets import load_dataset, Dataset
from rouge import Rouge

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}


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

    def __init__(self, model_path: str, encoder_name: str, sim_encoder_name: str):
        self.model_path = model_path
        self.encoder_name = encoder_name
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.encoder = SentenceTransformer(encoder_name) if encoder_name else None
        self.sim_encoder = SentenceTransformer(sim_encoder_name) if sim_encoder_name else None
        self.generator = ParaphrasingPipeline(model=self.model, tokenizer=self.tokenizer)

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['>>>>', '<quora>', '<|endoftext|>']}
        )
        return tokenizer

    def _load_model(self):
        model = GPT2LMHeadModel.from_pretrained(self.model_path) # baseline
        # model = GPT2ParaphrasingLM.from_pretrained(self.model_path) # gpt2 paraphrasing gpt2-para
        model.eval()
        return model

    def generate(self, sentence: str, strategy: str = "sampling", filter_best: bool = False, k: int = 10):
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
        return outputs_sent, self._filter_best(sentence, outputs_sent) if filter_best else ''

    def _filter_best(self, sentence: str, output_sent: [str]):
        original_embedding = normalize(self.sim_encoder.encode([sentence]), axis=1)
        sentences = list(output_sent)
        embeddings = normalize(self.sim_encoder.encode(sentences), axis=1)
        sim = np.matmul(original_embedding, np.transpose(embeddings))
        sim = [_s if _s < 1.0 else 0 for _s in sim[0]]
        idx = np.argmax(sim)
        print(sim, sentence, idx, sentences, sentences[idx])
        return sentences[idx]

    def _sampling(self, sentence, min_len, max_len: int, k: int, embeddings):
        return self.generator(sentence if embeddings is None else '',
                              min_length=min_len,
                              max_length=max_len,
                              do_sample=True,
                              repetition_penalty=1.8,
                              add_special_tokens=True,
                              num_return_sequences=k,
                              temperature=0.8,
                              top_p=0.75
                              # sentence_embedding=embeddings # gpt2-para
                              )

    def _beam_search(self, sentence, min_len, max_len: int, k: int, embeddings):
        return self.generator(sentence if embeddings is None else '',
                              min_length=min_len,
                              max_length=max_len,
                              repetition_penalty=1.8,
                              # add_special_tokens=True,
                              num_return_sequences=k,
                              num_beams=5,
                              no_repeat_ngram_size=2,
                              early_stopping=True
                              # sentence_embedding = embeddings # gpt2para

                              )


def _process_text(text: str):
    text = text.replace("<br />", " ")
    # paraphrased: <quora>
    text = ''.join(['<quora>', text, ' paraphrased: ']) # quora
    # text = ' '.join(['<|endoftext|>', text])
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_file",
        type=str, required=True, help="The input evaluation data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        default='./output_dir',
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--model_type", default="gpt2", type=str, help="The model architecture to be fine-tuned.")
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
        "--model_sim_name",
        default="",
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default="./.cache",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)",
    )
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --data_file "
            "or remove the --do_eval argument."
        )

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.block_size <= 0:
        args.block_size = (
            tokenizer.max_len_single_sentence
        )  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    # model = model_class.from_pretrained(
    #     args.model_name_or_path,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    #     cache_dir=args.cache_dir if args.cache_dir else None,
    # )
    # model.to(args.device)

    logger.info("Evaluation parameters %s", args)

    generator = ParaphrasingGenerator(args.model_name_or_path, args.model_encoder_name, args.model_sim_name)
    with open(args.data_file, "r", encoding='utf-8') as f:
        lines = [line.rstrip() for line in f]
    rouge = Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2)

    bleu_2, bleu_4, rouge_1, rouge_2, rouge_l = 0, 0, 0, 0, 0

    with open('results/baseline.txt', "w", encoding='utf-8') as f:
        for line in tqdm.tqdm(lines):
            input, refs = line.split('\t')
            orginal_input = input
            refs = literal_eval(refs)
            # refs = [ref.lower() for ref in refs]
            input = _process_text(input)
            cands, the_best = generator.generate(input, strategy="beam_search", filter_best=True, k=5)
            print(input, the_best)
            print('\t'.join([orginal_input, str(the_best), str(cands), str(refs)]), file=f)
            # cand = cand.split('>>>>')[1].split('?')[0] + '?'
            # cand = [_c.lower() for _c in cand]
        #     cand = cand[0].lower()
        #
        #     bleu_2 += sentence_bleu([ref.split() for ref in refs], cand.split(), weights=[0.5, 0.5, 0, 0])
        #     bleu_4 += sentence_bleu([ref.split() for ref in refs], cand.split())
        #     rouge_score = rouge.get_scores(cand, refs)
        #     # print(input, ' -> ', cand)
        #     # print(input, pred)
        #     rouge_1 += rouge_score['rouge-1']['f']
        #     rouge_2 += rouge_score['rouge-2']['f']
        #     rouge_l += rouge_score['rouge-l']['f']
        # n = len(lines[:100])
        # print(bleu_2 / n, bleu_4 / n, rouge_1 / n, rouge_2 / n, rouge_l / n)

        # evals
        # rouge = Rouge(metrics=['rouge-l', 'rouge-n'], max_n=2)
        # r1, r2, rL, rLsum, bleuall, bleuall_ori = 0, 0, 0, 0, 0, 0
        # for _i, sentences in tqdm.tqdm(enumerate(raw_datasets["validation"]["questions"], 1)):
        #     input = _process_text(sentences['text'][0])
        #     ref = _process_text(sentences['text'][1])
        #     pred = generator.generate(input, strategy="beam_search", filter_best=False, k=5)
        #
        #     max = 0
        #     maxr1 = 0
        #     maxr2 = 0
        #     bleu_ref_max = 0
        #     bleu_ori_max = 0
        #     bleu_ref = 0
        #     chencherry = SmoothingFunction()
        #     choosen_pred = ''
        #     # bleu_ori = bleu_score([[input.split(' ')]], [_pred.split(' ') for _pred in pred])
        #     # for _pred in pred:
        #     #     # results = rouge.get_scores(_pred, input)
        #     #     # bleu_ori = nltk.translate.bleu_score.sentence_bleu([input.split(' ')], _pred.split(' '))
        #     #     bleu_ori = bleu_score([[input.split(' ')]], [_pred.split(' ')])
        #     #
        #     #     # _r1 = results["rouge-1"]['f']
        #     #     # _r2 = results["rouge-2"]['f']
        #     #     # _rL = results["rouge-l"]['f']
        #     #     # if max < _rL:
        #     #     #     max = _rL
        #     #     # if maxr1 < _r1:
        #     #     #     maxr1 = _r1
        #     #     # if maxr2 < _r2:
        #     #     #     maxr2 = _r2
        #     #     if bleu_ori_max < bleu_ori < 0.95:
        #     #         # c
        #     #         bleu_ref = bleu_score([[ref.split(' ')]], [_pred.split(' ')])
        #     #         bleu_ori_max = bleu_ori
        #     #         choosen_pred = _pred
        #     # # rL += max
        #     # r1 += maxr1
        #     # r2 += maxr2
        #
        #     rouge.get_scores()
        #     bleuall += bleu_ref
        #     bleuall_ori += bleu_ori_max
        #     print('##Input ' + sentences['text'][0], '\n##REF:', ref, "\n##GENERATED: " + choosen_pred,
        #           f'[BLEU-ori: {bleu_ori_max}, BLEU-ref: {bleu_ref}]')
        #     print(f'[BLEU-ori: {bleuall_ori / _i}, BLEU-ref: {bleuall / (_i)}]')
        #     # rLsum += results["rougeLsum"]['f']
        #     # print('rouge1: ', results["rouge1"].mid.fmeasure)
        #     # print('rouge2: ', results["rouge2"].mid.fmeasure)
        #     # print('rougeL: ', results["rougeL"].mid.fmeasure)
        #     # print('rougeLsum: ', results["rougeLsum"].mid.fmeasure)
        # div = len(raw_datasets["validation"]["questions"])
        # print('rouge1: ', r1 / div)
        # print('rouge2: ', r2 / div)
        # print('rougeL: ', rL / div)
        # print('bleu: ', bleuall / div)
        # print('rougeLsum: ', rLsum / div)
