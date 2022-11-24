from ast import literal_eval
from typing import Set

import numpy as np
import tqdm
from sklearn.preprocessing import normalize
from BackTranslation import BackTranslation
from sentence_transformers import SentenceTransformer

from util import filter_best

encoder_name = "paraphrase-multilingual-mpnet-base-v2"


class BackTranslationGenerator:

    def __init__(self):
        self.trans = BackTranslation()
        self.encoder = SentenceTransformer(encoder_name)

    def translate(self, input, tmp):
        try:
            return self.trans.translate(input, src='en', tmp=tmp).result_text
        except Exception as ex:
            print(f'WARN: {ex}')
            return ''

    def generate(self, input):
        result0 = self.translate(input, tmp='zh-cn')
        result1 = self.translate(input, tmp='de')
        result2 = self.translate(input, tmp='fr')
        result3 = self.translate(input, tmp='pl')
        result4 = self.translate(input, tmp='ca')
        return [result0, result1, result2, result3, result4], filter_best(input, [result0, result1, result2, result3, result4])


if __name__ == '__main__':
    generator: BackTranslationGenerator = BackTranslationGenerator()
    with open('../.data/quora/test.txt', "r", encoding='utf-8') as f:
        lines = [line.rstrip() for line in f]
    with open('results/back_transl.txt', "w", encoding='utf-8') as f:
        for line in tqdm.tqdm(lines):
            input, refs = line.split('\t')
            orginal_input = input
            refs = literal_eval(refs)
            refs = [ref.lower() for ref in refs]
            cands, the_best = generator.generate(input)
            print(input, the_best)
            print('\t'.join([orginal_input, str(the_best), str(cands), str(refs)]), file=f)
