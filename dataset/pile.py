from typing import List

import torch
from datasets import load_dataset, Dataset, DatasetDict
from tokenizers.pre_tokenizers import Digits

import config
from tokenizers import pre_tokenizers, Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class LanguageModelDataset:

    def __init__(self, path: [str, List[str]], tokenizer=None, sentence_transformer=None):
        self.path = path
        self.tokenizer: Tokenizer = tokenizer
        self.sentence_transformer = sentence_transformer
        dataset: Dataset = load_dataset(self.path, split="train", cache_dir=f'{config.data_path}')
        dataset.cleanup_cache_files()
        train_test = dataset.train_test_split(test_size=0.1, seed=42)
        test_valid = train_test['test'].train_test_split(test_size=0.5)
        self.dataset = DatasetDict({
            'train': train_test['train'],
            'test': test_valid['test'],
            'valid': test_valid['train']})

    def build(self):
        for key in self.dataset:
            self.dataset[key]: Dataset = self.dataset[key].map(self.tokenize, batch_size=256, batched=True)
            self.dataset[key].set_format(type='torch', columns=['input_ids', 'target_ids'],
                                         output_all_columns=True, device=device)
            # self.dataset[key] = self.dataset[key].map(self.transform, batch_size=1024, batched=True)
            # self.dataset[key].set_format(type='torch', columns=['text', 'input_ids', 'target_ids'],
            #                              output_all_columns=True, device=device)

    def tokenize(self, batch):
        # text = [text[:32] if len(text) > 32 else text for text in batch["text"]]
        tokenized = self.tokenizer(batch["text"], truncation=True, padding=True, max_length=64).data['input_ids']
        # tokenized = [encode for encode in self.tokenizer(batch["text"], truncation=True, padding=True, max_length=32)]
        target = [example[1:] + [0] for example in tokenized]

        return {'text': batch['text'], 'input_ids': tokenized, "target_ids": target}

    def build_tokenizer(self):
        tokenizer: BertWordPieceTokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=True,
            lowercase=True)
        tokenizer.enable_padding(pad_id=0, pad_type_id=0, pad_token="[PAD]", length=64)
        tokenizer.enable_truncation(max_length=64)

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([tokenizer.pre_tokenizer, Digits(individual_digits=True)])

        def batch_iterator():
            batch_length = 1024
            for i in range(0, len(self.dataset), batch_length):
                print(i / len(self.dataset))
                yield self.dataset[i: i + batch_length]["text"]

        tokenizer.train_from_iterator(batch_iterator(), vocab_size=30_000, min_frequency=2, limit_alphabet=1000,
                                      wordpieces_prefix='##',
                                      special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
        tokenizer.save_model("./bert-tokenizer")
        self.tokenizer = tokenizer

    def transform(self, batch):
        embeddings = self.sentence_transformer.encode(batch['text'],
                                                      convert_to_numpy=False,
                                                      convert_to_tensor=False)
        return {'text': embeddings}
