import json
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
import hashlib

import config


class ParaCorpus(ABC):

    def __init__(self, name, split, text_columns, label_column=None, seed=42):
        self.name = name
        self.split = split
        self.text_columns = text_columns
        self.label_column = label_column
        self.id2sentence = {}
        random.seed(seed)

    def generate(self, valid_size, test_size):
        dataset = load_dataset(self.name)[self.split]

        input2refs = defaultdict(lambda: [])
        all_sentences = set()

        # 1. generate id2sentence file for all unique sentences
        for row in tqdm(dataset):
            para_pair, is_paraphrase = self.read_para_pair(row)
            all_sentences.add(para_pair[0])
            all_sentences.add(para_pair[1])

            idx1, idx2 = self.read_ids(row)
            self.id2sentence[idx1] = para_pair[0]
            self.id2sentence[idx2] = para_pair[1]

        output_dir = Path(f'../{config.cache_path}/{self.name}/_tmp')
        output_dir.mkdir(exist_ok=True, parents=True)

        with open(f'../{config.cache_path}/{self.name}/_tmp/id2sentence.json', 'w+', encoding='utf8') as fp:
            json.dump(self.id2sentence, fp)

        # 2. generate id2id for paraphrases
        for row in tqdm(dataset):
            para_pair, is_paraphrase = self.read_para_pair(row)
            if is_paraphrase:
                idx1, idx2 = self.read_ids(row)
                input2refs[idx1].append(idx2)

        with open(f'../{config.cache_path}/{self.name}/_tmp/paraphrases_ids.json', "w+", encoding='utf8') as fp:
            json.dump(input2refs, fp)

        train_keys, valid_keys = self.split_train_test(input2refs, valid_size)
        train_keys, test_keys = self.split_train_test(train_keys, test_size)

        # 3. generate test.txt file
        with open(f'../{config.cache_path}/{self.name}/test.txt', "w+", encoding='utf-8') as fp:
            for key in test_keys:
                print('\t'.join([self.id2sentence[key], str([self.id2sentence[id] for id in input2refs[key]])]), file=fp)
                try:
                    # remove sentences if they are in test split
                    all_sentences.remove(self.id2sentence[key])
                    for ref in input2refs[key]:
                        all_sentences.remove(self.id2sentence[ref])
                except:
                    pass

        # 3. generate dev.txt file
        with open(f'../{config.cache_path}/{self.name}/dev.txt', "w+", encoding='utf-8') as f_dev:
            for key in valid_keys:
                for value_key in input2refs[key]:
                    print('\t'.join([self.id2sentence[key], self.id2sentence[value_key]]), file=f_dev)
                    try:
                        all_sentences.remove([self.id2sentence[key]])
                        for id in input2refs[key]:
                            all_sentences.remove([self.id2sentence[id]])
                    except:
                        pass

        with open(f'../{config.cache_path}/{self.name}/train.txt', "w+", encoding='utf-8') as f_train:
            for key in train_keys:
                for value_key in input2refs[key]:
                    print('\t'.join([self.id2sentence[key], self.id2sentence[value_key]]), file=f_train)

        with open(f'../{config.cache_path}/{self.name}/unsupervised.txt', "w+", encoding='utf-8') as fp:
            for sentence in all_sentences:
                print(sentence, file=fp)

    @abstractmethod
    def read_para_pair(self, row):
        raise NotImplementedError

    def read_ids(self, row):
        para_pair, _ = self.read_para_pair(row)
        return hashlib.sha1(para_pair[0].encode("utf-8")).hexdigest(), hashlib.sha1(para_pair[1].encode("utf-8")).hexdigest()

    def split_train_test(self, ids, test_n):
        test_keys = random.sample(list(ids), test_n)
        train_keys = [k for k in tqdm(ids) if k not in test_keys]
        return train_keys, test_keys

