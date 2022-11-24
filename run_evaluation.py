import argparse
import ast
from collections import defaultdict
from os import listdir
from os.path import join, isfile
from typing import Callable, Union
from tabulate import tabulate

import datasets
import numpy as np
import tqdm
from datasets import Metric
from sentence_transformers import SentenceTransformer

import config
import warnings
import logging

from numpy import mean
from sklearn.preprocessing import normalize

from util import filter_best

# noise remove
warnings.filterwarnings("ignore")


class Evaluator:

    def __init__(self, metrics: dict):
        self.metrics: {Metric: Union[(Callable, {}), Callable]} = {}
        for metric, getter in metrics.items():
            self.metrics[datasets.load_metric(metric)] = getter

    def compute(self, predictions, references):
        metrics = {}
        for metric, getter in self.metrics.items():
            kwargs = {}
            if type(getter) is tuple:
                kwargs = getter[1]
                getter = getter[0]
            try:
                metrics.update(getter(metric.compute(predictions=predictions, references=references, **kwargs)))
            except Exception as ex:
                print(ex)
                pass
        return metrics


def split(line):
    return line.split('\t')


def iblue(alpha, bleu, selfbleu):
    return bleu - (alpha * selfbleu)


def bert_iblue(beta, bertscore, selfbleu):
    return (beta + 1.0) / (beta * (1.0 / bertscore) + 1.0 * (1.0 / (1.0 - selfbleu)))


def print_toy(lines):
    print('> Examples of values [make sure if sentences are correct]:')
    for line in lines[:3]:
        input, best, generated, refs = split(line.strip())
        print('\tinput: ' + input)
        print('\tbest: ' + best)
        print('\tgenerated: ' + str(generated))
        print('\trefs:' + refs + '\n')


def selfBLEU(generated):
    metric = datasets.load_metric('bleu')
    selfbleu_acc = []
    if len(generated) < 2:
        logging.info('More than 2 generated sentences are needed to calculate the selfBLEU metric!')
        return 0
    for generated_sentence in generated:
        leftover = generated.copy()
        leftover.remove(generated_sentence)
        selfbleu = metric.compute(predictions=[generated_sentence.split()],
                                  references=[[sentence.split() for sentence in leftover]])
        selfbleu_acc.append(selfbleu.get('bleu'))
    if len(selfbleu_acc) == 0:
        logging.info('selfBLEU is empty.')
    return mean(selfbleu_acc)


def preprocess(text):
    return text.lower()


def run(args):

    refs_metrics = Evaluator(
        metrics={
            'bleu': lambda x: {'bleu': x['bleu']},
            'rouge': lambda x: {'rouge1': x['rouge1'].mid.fmeasure, 'rouge2': x['rouge2'].mid.fmeasure,
                                'rougeL': x['rougeL'].mid.fmeasure},
            'meteor': lambda x: {'meteor': x['meteor']},
            'bertscore': (lambda x: {'bertscore': x['f1'][0]}, {'lang': 'en'})
        })
    original_metrics = Evaluator(
        metrics={
            'bleu': lambda x: {'bleu': x['bleu']},
        })

    path = config.results_dir_path
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        refs_scores = defaultdict(lambda x: [])
        original_scores = defaultdict(lambda x: [])
        sim_input_gen = []
        sim_best_refs = []
        with open(file, 'r', encoding='utf8') as file:
            lines = file.readlines()

            if args.debug_mode:
                lines = lines[:10]

            if args.print_toy_examples:
                print_toy(lines)

            print(f'> Start evaluation for {file.name} results [{len(lines)}]\n')

            selfbleu_global = []
            for i, line in tqdm.tqdm(enumerate(lines, 1)):
                if not line:
                    logging.info(f'WARN: The empty line ({i}) detected!')
                    continue
                try:
                    input, best, generated, refs = (preprocess(text) for text in split(line.strip()))
                except Exception as e:
                    logging.info(f'Line {i}: {e}')
                    if args.ignore_exceptions:
                        continue
                    else:
                        break

                generated = ast.literal_eval(generated)
                if '' in generated:
                    logging.info(f'WARN: Empty values in generated sentences in line{i}.')
                    generated = list(filter(lambda a: a != '', generated))

                if best == config.unk_token:
                    best = filter_best(input, generated, encoder)
                refs = ast.literal_eval(refs)

                scores = refs_metrics.compute(predictions=[best.split()], references=[[ref.split() for ref in refs]])
                ori_scores = original_metrics.compute(predictions=[best.split()], references=[[ref.split() for ref in refs]])

                selfbleu_global.append(selfBLEU(generated))

                refs_scores = {
                    key: refs_scores.get(key, []) + [scores.get(key, [])] for key in set(refs_scores) | set(scores)
                }
                original_scores = {
                    key: original_scores.get(key, []) + [ori_scores.get(key, [])] for key in set(original_scores) | set(ori_scores)
                }

                ### sent-trans measure: input -> generated
                original_embedding = normalize(encoder.encode([input]), axis=1)
                sentences = list(generated)
                embeddings = normalize(encoder.encode(sentences), axis=1)
                sim_input_gen.append(mean(np.matmul(original_embedding, np.transpose(embeddings))))

                ### sent-trans measure: best -> refs
                original_embedding = normalize(encoder.encode([best]), axis=1)
                sentences = list(refs)
                embeddings = normalize(encoder.encode(sentences), axis=1)
                sim_best_refs.append(mean(np.matmul(original_embedding, np.transpose(embeddings))))

            print(file.name, end=': \n')

            print('\n\t' + 'ori-bleu', end=' | ')
            print('\tself-bleu', end=' | ')
            print('\tbleu', end=' | ')
            print('\trouge1', end=' | ')
            print('\trouge2', end=' | ')
            print('\trougeL', end=' | ')
            print('\tmeteor', end=' | ')
            print('\tBERTScore', end=' | ')
            # print(str(round(mean(ibleu_arr), 3)), end=' & ')
            # print(str(round(mean(bert_ibleu_arr), 3)), end=' & ')
            print('\tsim_input_gens', end=' | ')
            print('\tsim_best_refs', end=' \n ')

            print('\t'+str(round(mean(original_scores['bleu']) * 100, 2)), end=' & ')
            print('\t\t'+str(round(mean(selfbleu_global) * 100, 2)), end=' & ')
            print('\t\t'+str(round(mean(np.array(refs_scores['bleu'])) * 100, 2)), end=' & ')
            print('\t\t'+str(round(mean(np.array(refs_scores['rouge1'])) * 100, 2)), end=' & ')
            print('\t'+str(round(mean(np.array(refs_scores['rouge2'])) * 100, 2)), end=' & ')
            print('\t'+str(round(mean(np.array(refs_scores['rougeL'])) * 100, 2)), end=' & ')
            print('\t'+str(round(mean(np.array(refs_scores['meteor'])) * 100, 2)), end=' & ')
            print('\t'+str(round(mean(np.array(refs_scores['bertscore'])) * 100, 2)), end=' & ')
            # print(str(round(mean(ibleu_arr), 3)), end=' & ')
            # print(str(round(mean(bert_ibleu_arr), 3)), end=' & ')
            print('\t'+str(round(mean(sim_input_gen) * 100, 2)), end=' & ')
            print('\t'+str(round(mean(sim_best_refs) * 100, 2)), end=' \n\n')

            # print({k: round(v / i, 3) for k, v in scores.items()})
            # print('ibleu: ', {k: round(v / i, 3) for k, v in scores_ori.items()})
            # print('slef-bleu: ', str(mean(selfbleu_all_all)))
            # print('ibleu: ' + str(round(mean(np.array(ibleu_arr)), 3)))
            # print('bert_ibleu: ' + str(round(mean(np.array(bert_ibleu_arr)), 3)))

    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--sentences_similarity_encoder', required=False, default='paraphrase-mpnet-base-v2')

    parser.add_argument('--ignore_exceptions', required=False, default=False)
    parser.add_argument('--print_toy_examples', required=False, default=True)
    parser.add_argument('--debug_mode', required=False, action='store_true', default=False)

    args = parser.parse_args()
    run(args)
