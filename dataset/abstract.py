import json
import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import hashlib

import config


class ParaCorpus(ABC):

    def __init__(self, name, splits, text_columns, label_column=None, seed=42, test_split=None, configs=None):
        self.name = name
        self.splits = splits
        self.text_columns = text_columns
        self.label_column = label_column
        self.id2sentence = {}
        self.test_split = test_split
        self.configs = configs
        random.seed(seed)

    def generate(self, valid_size, test_size, output_dir=None, concatenate_splits=False):
        cache_path = output_dir if output_dir else f'{config.data_path}/{self.name}'

        dataset = load_dataset(self.name, self.configs if config else None)
        if self.splits:
            dataset = [dataset[split] for split in self.splits]
        if concatenate_splits:
            dataset = concatenate_datasets(dataset)

        input2refs = defaultdict(lambda: [])
        all_sentences = set()

        # 1. generate id2sentence file for all unique sentences
        # print( type(dataset))
        # if type(dataset) == list:
        # for split in
        for row in tqdm(dataset[0]):
            para_pair, is_paraphrase = self.read_para_pair(row)
            for item in para_pair:
                all_sentences.add(item)

            for idx, item in zip(self.read_ids(row), para_pair):
                self.id2sentence[idx] = item

        output_dir = Path(f'{cache_path}/_tmp')
        output_dir.mkdir(exist_ok=True, parents=True)

        with open(f'{cache_path}/_tmp/id2sentence.json', 'w+', encoding='utf8') as fp:
            json.dump(self.id2sentence, fp)

        # 2. generate id2id for paraphrases
        for row in tqdm(dataset[0]):
            para_pair, is_paraphrase = self.read_para_pair(row)
            if is_paraphrase:
                idx1, idx2 = self.read_ids(row)
                input2refs[idx1].append(idx2)

        with open(f'{cache_path}/_tmp/paraphrases_ids.json', "w+", encoding='utf8') as fp:
            json.dump(input2refs, fp)

        if valid_size:
            train_keys, valid_keys = self.split_train_test(input2refs, valid_size)
        else:
            pass

        if test_size:
            train_keys, test_keys = self.split_train_test(train_keys, test_size)
        else:
            pass

        # 3. generate test.txt file
        with open(f'{cache_path}/test.tsv', "w+", encoding='utf-8') as fp:
            if not self.test_split:
                for key in test_keys:
                    print('\t'.join([self.id2sentence[key], str([self.id2sentence[id] for id in input2refs[key]])]),
                          file=fp)
                    try:
                        # remove sentences if they are in test split
                        all_sentences.remove(self.id2sentence[key])
                        for ref in input2refs[key]:
                            all_sentences.remove(self.id2sentence[ref])
                    except:
                        pass

        # 3. generate dev.txt file
        with open(f'{cache_path}/dev.tsv', "w+", encoding='utf-8') as f_dev:
            for key in valid_keys:
                for value_key in input2refs[key]:
                    print('\t'.join([self.id2sentence[key], self.id2sentence[value_key]]), file=f_dev)
                    try:
                        all_sentences.remove([self.id2sentence[key]])
                        for id in input2refs[key]:
                            all_sentences.remove([self.id2sentence[id]])
                    except:
                        pass

        with open(f'{cache_path}/train.tsv', "w+", encoding='utf-8') as f_train:
            for key in train_keys:
                for value_key in input2refs[key]:
                    print('\t'.join([self.id2sentence[key], self.id2sentence[value_key]]), file=f_train)

        with open(f'{cache_path}/unsupervised.txt', "w+", encoding='utf-8') as fp:
            for sentence in all_sentences:
                print(sentence, file=fp)

        print(f'train.tsv, valid.tsv, test.tsc, unsupervised.txt files saved in {cache_path}')

    @abstractmethod
    def read_para_pair(self, row):
        raise NotImplementedError

    def read_ids(self, row):
        para_pair, _ = self.read_para_pair(row)
        return hashlib.sha1(para_pair[0].encode("utf-8")).hexdigest(), hashlib.sha1(
            para_pair[1].encode("utf-8")).hexdigest()

    def split_train_test(self, url2texts, test_n):
        test_keys = random.sample(list(url2texts), test_n)
        train_keys = [k for k in tqdm(url2texts) if k not in test_keys]
        return train_keys, test_keys
