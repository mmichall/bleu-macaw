from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from abstract import ParaCorpus
import config


class MSCOCOParaCorpus(ParaCorpus):

    def __init__(self, seed):
        super().__init__('ChristophSchuhmann/MS_COCO_2017_URL_TEXT', ['train'], ['TEXT'], id, seed=seed)

    def read_para_pair(self, row):
        return row['URL'], row['TEXT'], 1

    def read_ids(self, row):
        return row[self.text_columns[0]]['URL']

    def generate(self, valid_size, test_size, output_dir=None):
        cache_path = output_dir if output_dir else f'{config.data_path}/{self.name}'
        dataset = load_dataset(self.name)[self.splits[0]]

        url2texts = defaultdict(lambda: [])
        all_sentences = set()

        # 1. generate id2sentence file for all unique sentences
        # input2refs = {}
        for row in tqdm(dataset):
            url, text = self.read_para_pair(row)
            all_sentences.add(text)
            url2texts[url].append(text)

        output_dir = Path(f'{cache_path}/_tmp')
        output_dir.mkdir(exist_ok=True, parents=True)

        train_keys, valid_keys = self.split_train_test(url2texts, valid_size)
        train_keys, test_keys = self.split_train_test(train_keys, test_size)

        # 3. generate test.txt file
        with open(f'{cache_path}/test.tsv', "w+", encoding='utf-8') as fp:
            for key in test_keys:
                try:
                    for ref in url2texts[key]:
                        all_sentences.remove(ref)
                except:
                    pass
                random_index = np.random.randint(0, len(url2texts[key]))
                input = url2texts[key].pop(random_index)
                print('\t'.join([input, str(url2texts[key])]), file=fp)

        # 3. generate dev.txt file
        with open(f'{cache_path}/dev.tsv', "w+", encoding='utf-8') as f_dev:
            for key in valid_keys:
                try:
                    for ref in url2texts[key]:
                        all_sentences.remove(ref)
                except:
                    pass
                random_index = np.random.randint(0, len(url2texts[key]))
                input = url2texts[key].pop(random_index)
                for ref in url2texts[key]:
                    print('\t'.join([input, ref]), file=f_dev)

        with open(f'{cache_path}/train.tsv', "w+", encoding='utf-8') as f_train:
            for key in train_keys:
                try:
                    for ref in url2texts[key]:
                        all_sentences.remove(ref)
                except:
                    pass
                random_index = np.random.randint(0, len(url2texts[key]))
                input = url2texts[key].pop(random_index)
                for ref in url2texts[key]:
                    print('\t'.join([input, ref]), file=f_train)

        with open(f'{cache_path}/unsupervised.txt', "w+", encoding='utf-8') as fp:
            for sentence in all_sentences:
                print(sentence, file=fp)

        print(f'train.tsv, valid.tsv, test.tsc, unsupervised.txt files saved in {cache_path}')
