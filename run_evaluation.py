import argparse
import ast
import random
import string
from collections import defaultdict
from os import listdir
from os.path import join, isfile
from typing import Callable, Union
from sentence_transformers import util

import datasets
import numpy as np
import pandas as pd
import tqdm
from datasets import Metric
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

import config
import warnings
import logging

from numpy import mean
from sklearn.preprocessing import normalize

from util import filter_best
import torch

warnings.filterwarnings("ignore")


def patch_sentence_transformer(model: SentenceTransformer):
    try:
        from optimum.bettertransformer import BetterTransformer
        from sentence_transformers.models import Transformer
        for module in model.modules():
            if isinstance(module, Transformer):
                module.auto_model = BetterTransformer.transform(module.auto_model)
                return True
    except (ImportError, Exception):
        pass
    return False


class Adequacy():

    def __init__(self, model_tag='prithivida/parrot_adequacy_model'):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.adequacy_model = AutoModelForSequenceClassification.from_pretrained(model_tag)
        self.tokenizer = AutoTokenizer.from_pretrained(model_tag)

    def score(self, input_phrase, para_phrases, adequacy_threshold=0, device="cuda"):
        adequacy_scores = {}
        for para_phrase in para_phrases:
            x = self.tokenizer(input_phrase, para_phrase, return_tensors='pt', max_length=128, truncation=True)
            x = x.to(device)
            self.adequacy_model = self.adequacy_model.to(device)
            logits = self.adequacy_model(**x).logits
            probs = logits.softmax(dim=1)
            prob_label_is_true = probs[:, 1]
            adequacy_score = prob_label_is_true.item()
            if adequacy_score >= adequacy_threshold:
                adequacy_scores[para_phrase] = adequacy_score
        return adequacy_scores


class Fluency():
    def __init__(self, model_tag='prithivida/parrot_fluency_model'):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.fluency_model = AutoModelForSequenceClassification.from_pretrained(model_tag, num_labels=2)
        self.fluency_tokenizer = AutoTokenizer.from_pretrained(model_tag)


    def score(self, para_phrases, fluency_threshold=0, device="cuda"):
        from scipy.special import softmax
        self.fluency_model = self.fluency_model.to(device)
        fluency_scores = {}
        for para_phrase in para_phrases:
            input_ids = self.fluency_tokenizer("Sentence: " + para_phrase, return_tensors='pt', truncation=True)
            input_ids = input_ids.to(device)
            prediction = self.fluency_model(**input_ids)
            scores = prediction[0][0].detach().cpu().numpy()
            scores = softmax(scores)
            fluency_score = scores[1]  # LABEL_0 = Bad Fluency, LABEL_1 = Good Fluency
            if fluency_score >= fluency_threshold:
                fluency_scores[para_phrase] = fluency_score
        return fluency_scores


# prithivida / parrot_paraphraser_on_T5
# adequacy_score = Adequacy()
fluency_score = Fluency()

bertscore = datasets.load_metric("bertscore")

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
                logging.info(ex)
                continue
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
        best = config.unk_token
        print('\tinput: ' + input)
        print('\tbest: ' + best)
        print('\tgenerated: ' + str(generated))
        print('\trefs:' + refs + '\n')


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')


bleu_metric = datasets.load_metric('bleu')
def selfBLEU(generated):
    selfbleu_acc = []
    if len(generated) < 2:
        logging.info('More than 2 generated sentences are needed to calculate the selfBLEU metric!')
        return 0
    for generated_sentence in generated:
        leftover = generated.copy()
        leftover.remove(generated_sentence)
        try:
            selfbleu = bleu_metric.compute(predictions=[generated_sentence.lower().split()],
                                      references=[[sentence.lower().split() for sentence in leftover]], max_order=3)
            selfbleu_acc.append(selfbleu.get('bleu'))
        except:
            selfbleu_acc.append(0)
    if len(selfbleu_acc) == 0:
        logging.info('selfBLEU is empty.')
    return mean(selfbleu_acc)


def preprocess(text):
    return ' '.join(word_tokenize(text.lower()))

# bertscore =
def run(args):
    encoder = SentenceTransformer(args.sentences_encoder)
    encoder.eval()
    encoder.half()
    patch_sentence_transformer(encoder)

    refs_metrics = Evaluator(
        metrics={
            'bleu': (lambda x: {'bleu': x['bleu']}, {'max_order': 3}),
            'rouge': lambda x: {#'rouge1': x['rouge1'].mid.fmeasure, 'rouge2': x['rouge2'].mid.fmeasure,
                                'rougeL': x['rougeL'].mid.fmeasure}
        })
    refs_metrics_sim = Evaluator(
        metrics={
            'bertscore': (
            lambda x: {'bertscore': x['f1'][0]}, {'lang': 'en', 'model_type': 'microsoft/deberta-large-mnli'})
        })
    original_metrics = Evaluator(
        metrics={
            'bleu': (lambda x: {'bleu': x['bleu']}, {'max_order': 3})
        })
    original_metrics_sim = Evaluator(
        metrics={
            'bertscore': (
            lambda x: {'bertscore': x['f1'][0]}, {'lang': 'en', 'model_type': 'microsoft/deberta-large-mnli'})
            # deberta-large-mnli (faster) or deberta-xlarge-mnli
        })

    path = config.results_dir_path + '/cnn_news'
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    iii = 0
    for file in files:
        refs_scores = defaultdict(lambda: [])
        original_scores = defaultdict(lambda: [])
        refs_scores_sim = defaultdict(lambda: [])
        original_scores_sim = defaultdict(lambda: [])
        sim_input_gen = []
        sim_best_refs = []
        with open(file, 'r', encoding='utf8') as file:
            lines = file.readlines()

            if args.debug_mode:
                lines = lines[:10]

            # if args.print_toy_examples:
            #     print_toy(lines)

            print(f'> Start evaluation for {file.name} [{len(lines)}]\n')

            selfbleu_global = []
            bert_ibleu = []
            sbert_ibleu = []
            sbert_ibleu_cands = []
            adequacy = []
            fluency = []

            _inputs = []
            _bests = []
            _generated = []
            _refs = []
            for i, line in tqdm.tqdm(enumerate(lines, 1)):
                if not line:
                    logging.info(f'WARN: The empty line ({i}) detected!')
                    continue
                try:
                    input, best, generated, refs = line.split('\t')
                    # input = input.lower()
                except Exception as e:
                    logging.info(f'Line {i}: {e}')
                    if args.ignore_exceptions:
                        continue
                    else:
                        break
                _inputs.append(input)
                generated = list(ast.literal_eval(generated))
                if '' in generated:
                    logging.info(f'WARN: Empty values in generated sentences in line {i}.')
                    generated = list(filter(lambda a: a != '', generated))
                if len(generated) > 5:
                    generated = random.choices(generated, k=5)
                if len(generated) == 0:
                    generated = [input]
                # generated = list(map(lambda sent: sent.lower(), generated))
                _generated.append(generated)
                if len(generated) > 0 and best == '###':
                    best = filter_best(input, generated)
                # else:
                #     best = '---'
                # best = best.lower()
                _bests.append(best)
                _refs.append(list(map(lambda sent: sent, ast.literal_eval(refs))))

            with torch.no_grad():
                _inputs_embedding = encoder.encode(_inputs, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
                _bests_embedding = encoder.encode(_bests, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

            for input, input_embedd, best_embedd, generated, best, refs in tqdm.tqdm(zip(_inputs, _inputs_embedding, _bests_embedding, _generated, _bests, _refs)):
                norm_input = input.translate(str.maketrans('', '', string.punctuation)).lower()
                norm_best = best.translate(str.maketrans('', '', string.punctuation)).lower()
                norm_refs = [ref.translate(str.maketrans('', '', string.punctuation)).lower() for ref in refs]
                norm_gens = [gen.translate(str.maketrans('', '', string.punctuation)).lower() for gen in generated]

                scores = refs_metrics.compute(predictions=[norm_best.split()], references=[[ref.split() for ref in norm_refs]])
                scores_sim = refs_metrics_sim.compute(predictions=[best.split()],
                                              references=[[ref.split() for ref in refs]])
                # scores_all = refs_metrics.compute(predictions=[[gen.split() for gen in generated]], references=[[ref.split() for ref in refs]])
                ori_scores = original_metrics.compute(predictions=[norm_input.split()], references=[[gen.split() for gen in norm_gens]])
                ori_scores_sim = original_metrics_sim.compute(predictions=[input.split()],
                                                      references=[[gen.split() for gen in generated]])

                selfbleu_global.append(selfBLEU(norm_gens))
                # sprawdzić alpha i beta
                # bert_ibleu_global.append(bert_iblue(generated))
                # print(mean(list(adequacy_score.score(input, generated).values())))
                # adequacy.append(mean(list(adequacy_score.score(best, refs).values())))
                # print(fluency_score.score(generated))
                fluency.append(mean(list(fluency_score.score(generated).values())))

                refs_scores = {
                    key: refs_scores.get(key, []) + [scores.get(key, 0)] for key in set(refs_scores) | set(scores)
                }
                original_scores = {
                    key: original_scores.get(key, []) + [ori_scores.get(key, 0)] for key in set(original_scores) | set(ori_scores)
                }

                refs_scores_sim = {
                    key: refs_scores_sim.get(key, []) + [scores_sim.get(key, 0)] for key in set(refs_scores_sim) | set(scores_sim)
                }
                original_scores_sim = {
                    key: original_scores_sim.get(key, []) + [ori_scores_sim.get(key, 0)] for key in
                    set(original_scores_sim) | set(ori_scores_sim)
                }

                ### sent-trans measure: input -> generated
                with torch.no_grad():
                    # original_embedding = encoder.encode([input], normalize_embeddings=True)
                    original_embedding = input_embedd
                sentences = list(generated)
                with torch.no_grad():
                    embeddings = encoder.encode(sentences, normalize_embeddings=True)
                cos_sim = util.cos_sim(original_embedding, embeddings).numpy()
                sim_input_gen.append(np.mean(cos_sim))
                # sim_input_gen_max.append(max(cos_sim[cos_sim != 1.0]))

                ### sent-trans measure: best -> refs
                #original_embedding = normalize(encoder.encode([best]), axis=1)
                with torch.no_grad():
                    # original_embedding = encoder.encode([best], normalize_embeddings=True)
                    original_embedding = best_embedd
                    # original_embedding = encoder.encode(generated, normalize_embeddings=True)
                sentences = list(refs)
                with torch.no_grad():
                    embeddings = encoder.encode(sentences, normalize_embeddings=True)
                # sim_best_refs.append(mean(np.matmul(original_embedding, np.transpose(embeddings))))
                # with torch.no_grad():
                #     embeddings = encoder.encode(sentences, normalize_embeddings=True)
                cos_sim = util.cos_sim(original_embedding, embeddings).numpy()  # sprawdzić poprawnośc
                sim_best_refs.append(np.mean(cos_sim))

                sim_input_best = np.mean(util.cos_sim(input_embedd, best_embedd).numpy())

                # print(ori_scores['bleu'])
                beta = 2.0
                epsilon = 0.00000001

                # agg = []
                # for gen in generated:
                #     results = bleu_metric.compute(predictions=[input.split()], references=[[gen.split()]], max_order=2)
                #     agg.append(results['bleu'])
                # print(np.mean(agg))
                # try:
                #     self_bleu_best = bleu_metric.compute(predictions=[input.lower().split()], references=[[best.lower().split()]], max_order=3) # może jeszcze bleu z wszystkimi cands?
                # except:
                #     self_bleu_best = {'bleu': 1.0}
                ###
                try:
                    self_bleu_gens = bleu_metric.compute(predictions=[norm_input.split()],
                                                         references=[[norm_best.split()]],
                                                         max_order=3)  # może jeszcze bleu z wszystkimi cands?
                except:
                    self_bleu_gens = {'bleu': 1.0}
                                      # self_bleu_cands = bleu_metric.compute(predictions=[input.lower().split()], references=[[cand.lower().split() for cand in generated]], max_order=3)  # może jeszcze bleu z wszystkimi cands?
                #
                # X = vectorizer.fit_transform([input, best]).toarray()
                # x_cos_sim = cosine_similarity(X)
                # print(np.max(cos_sim), self_bleu_best['bleu'])

                # print(round(ori_scores['bertscore'], 2),
                #       round(self_bleu_gens['bleu'], 2),
                #       round(pow((beta * (pow(ori_scores['bertscore'], -1)) + pow((1.+epsilon) - self_bleu_gens['bleu'], -1)) / (beta + 1.0), -1), 2),
                #       input.lower(),
                #       generated)
                # print(round(np.mean(cos_sim), 2),
                #       round(self_bleu_gens['bleu'], 2),
                #       round(pow((beta * (pow(np.mean(cos_sim), -1)) + pow((1.+epsilon) - self_bleu_gens['bleu'], -1)) / (beta + 1.0), -1), 2),
                #       input.lower(),
                #       generated)

                bs = bertscore.compute(predictions=[input.split()], references=[[best.split()]],
                                       lang='en', model_type='microsoft/deberta-large-mnli')

                beta = 2.0
                bert_ibleu.append(pow((beta * (pow(bs['f1'][0], -1)) + pow((1.+epsilon) - self_bleu_gens['bleu'], -1)) / (beta + 1.0), -1)) # ori_scores['bleu']
                # print(input, best, bs['f1'][0], self_bleu_gens['bleu'], pow((beta * (pow(bs['f1'][0], -1)) + pow((1.+epsilon) - self_bleu_gens['bleu'], -1)) / (beta + 1.0), -1))

                # beta = 4.0
                sbert_ibleu.append(pow((beta * (pow(sim_input_best, -1)) + pow((1.+epsilon) - self_bleu_gens['bleu'], -1)) / (beta + 1.0), -1))
                # sbert_ibleu_cands.append(pow((beta * (pow(np.mean(cos_sim), -1)) + pow((1. + epsilon) - self_bleu_cands['bleu'], -1)) / (beta + 1.0), -1))

            print(file.name, end=': \n')

            print('\n\t' + 'ori-bleu4', end=' | ')
            print('\tself-bleu4', end=' | ')
            print('\tbleu4', end=' | ')
            # print('\trouge1', end=' | ')
            # print('\trouge2', end=' | ')
            print('\trougeL', end=' | ')
            # print('\tmeteor', end=' | ')
            # print('\tadequacy', end=' | ')
            print('\tfluency', end=' | ')
            print('\tori-BERTScore', end=' | ')
            print('\tBERTScore', end=' | ')
            # print(str(round(mean(ibleu_arr), 3)), end=' & ')
            print('\tori-SBERT', end=' | ')
            print('\tSBERT', end=' | ')
            print('\tBERTScore-iBLEU2', end=' | ')
            print('\tSBERT-iBLEU2', end=' \n ')
            # print('\tSBERT-iBLEU2-cands', end=' \n ')

            print('\t'+str(round(mean(np.array(original_scores['bleu'])) * 100, 2)), end=' & ')
            print('\t\t'+str(round(mean(selfbleu_global) * 100, 2)), end=' & ')
            print('\t\t'+str(round(mean(np.array(refs_scores['bleu'])) * 100, 2)), end=' & ')
            # print('\t\t'+str(round(mean(np.array(refs_scores['rouge1'])) * 100, 2)), end=' & ')
            # print('\t'+str(round(mean(np.array(refs_scores['rouge2'])) * 100, 2)), end=' & ')
            print('\t'+str(round(mean(np.array(refs_scores['rougeL'])) * 100, 2)), end=' & ')
            # print('\t'+str(round(mean(np.array(refs_scores['meteor'])) * 100, 2)), end=' & ')
            # print('\t' + str(round(mean(adequacy) * 100, 2)), end=' & ')
            print('\t' + str(round(mean(fluency) * 100, 2)), end=' & ')
            print('\t' + str(round(mean(np.array(original_scores_sim['bertscore'])) * 100, 2)), end=' & ')
            print('\t'+str(round(mean(np.array(refs_scores_sim['bertscore'])) * 100, 2)), end=' & ')
            # print(str(round(mean(ibleu_arr), 3)), end=' & ')
            print('\t' + str(round(mean(sim_input_gen) * 100, 2)), end=' & ')
            print('\t' + str(round(mean(sim_best_refs) * 100, 2)), end=' & ')
            print('\t' + str(round(mean(bert_ibleu) * 100, 2)), end=' & ')
            print('\t' + str(round(mean(sbert_ibleu) * 100, 2)), end=' \n ')
            # print('\t' + str(round(mean(sbert_ibleu_cands) * 100, 2)), end=' \n ')


            # all generated to all references
            # print('\t'+'---', end=' & ')
            # print('\t\t'+'---', end=' & ')
            # print('\t\t'+str(round(mean(np.array(refs_all_scores['bleu'])) * 100, 2)), end=' & ')
            # # print('\t\t'+str(round(mean(np.array(refs_scores['rouge1'])) * 100, 2)), end=' & ')
            # # print('\t'+str(round(mean(np.array(refs_scores['rouge2'])) * 100, 2)), end=' & ')
            # print('\t'+str(round(mean(np.array(refs_all_scores['rougeL'])) * 100, 2)), end=' & ')
            # print('\t'+str(round(mean(np.array(refs_all_scores['meteor'])) * 100, 2)), end=' & ')
            # print('\t'+str(round(mean(np.array(refs_all_scores['bertscore'])) * 100, 2)), end=' \n\n ')

            # print({k: round(v / i, 3) for k, v in scores.items()})
            # print('ibleu: ', {k: round(v / i, 3) for k, v in scores_ori.items()})
            # print('slef-bleu: ', str(mean(selfbleu_all_all)))
            # print('ibleu: ' + str(round(mean(np.array(ibleu_arr)), 3)))
            # print('bert_ibleu: ' + str(round(mean(np.array(bert_ibleu_arr)), 3)))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--sentences_encoder',
                        required=False,
                        default='sentence-transformers/paraphrase-distilroberta-base-v2')

    parser.add_argument('--ignore_exceptions', required=False, default=False)
    parser.add_argument('--print_toy_examples', required=False, default=True)
    parser.add_argument('--debug_mode', required=False, action='store_true', default=False)

    args = parser.parse_args()
    run(args)
