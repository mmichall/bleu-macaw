import os
from collections import Counter, defaultdict

import torch
import numpy as np
from pandas import Series
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import IMDB
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

from readers.reader import Reader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class LanguageModelingDataset(Dataset):
    def __init__(self, reader: Reader, vocab: Vocab = None, tokenizer=None, min_freq=None, sequence_length=100, uuid=None):
        self.reader = reader
        self.vocab: Vocab = vocab
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        if not vocab:
            path = f'.cache/vocab/{uuid}_{min_freq}.vb'
            if os.path.exists(path):
                self.vocab = torch.load(path)
                print(f'Vocabulary has been loaded from {path}')
            else:
                print('Build vocabulary...', end='')
                self.vocab = self.build_vocab(min_freq=min_freq)
                print('DONE')
                torch.save(self.vocab, path)
                print(f'Saved in {path}')
        self.vocab_len = len(self.vocab)
        self.stoi = self.vocab.get_stoi()
        self.ZEROS = torch.zeros(self.vocab_len, dtype=torch.int8)
        self.diagonal = torch.zeros(self.vocab_len, self.vocab_len, dtype=torch.int8)
        self.diagonal.fill_diagonal_(1)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        item: Series = self.reader.read_example(idx)
        tokenized = self.tokenizer(item['text'])
        _len = len(tokenized)
        if _len > self.sequence_length:
            tokenized = tokenized[:self.sequence_length]
        else:
            tokenized.extend(['<pad>'] * (self.sequence_length - _len))
        tokenized = torch.stack([self.ohe(self.stoi[token]) if token in self.stoi else self.ZEROS for token in tokenized])
        return tokenized

    def build_vocab(self, min_freq: int = 1) -> Vocab:
        texts: Series = self.reader.read_column('text')
        texts = texts.apply(self.tokenizer)
        vocab: Vocab = build_vocab_from_iterator(texts, min_freq=min_freq if min_freq else 1, specials=['<pad>'])
        vocab.set_default_index(0)
        return vocab

    def ohe(self, idx):
        return self.diagonal[idx]


def WikiText2DL(tokenizer_name='basic_english', most_common_words=None):
    tokenizer = get_tokenizer(tokenizer_name)

    vocab = None
    if most_common_words:
        train_dataset = IMDB(tokenizer=tokenizer, data_select='train')[0]
        vocab = build_vocab(train_dataset, most_common_words, f'wikitext2_{tokenizer_name}')

    train_dataset, test_dataset, valid_dataset = WikiText2(tokenizer=tokenizer, vocab=vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

    return train_dataloader, test_dataloader, valid_dataloader


def build_vocab(dataset: Dataset, most_common_words: int, cache_uid: str, rebuild=False):
    vocab_path = f'.cache/vocab/_{cache_uid}_{most_common_words}.vb'
    if os.path.exists(vocab_path) and not rebuild:
        print(f'Read from cache {vocab_path}')
        vocab = torch.load(vocab_path)
    else:
        print(f'Building {cache_uid} vocabulary...')
        counter = Counter(dict(dataset.vocab.freqs.most_common(most_common_words)))
        vocab = Vocab(counter)
        stoi = defaultdict(default_unk_index)
        for word in vocab.stoi:
            stoi[word] = len(stoi)
        vocab.stoi = stoi
        torch.save(vocab, vocab_path)
        print(f'Saved in {vocab_path}')
    return vocab


def default_unk_index():
    return Vocab.UNK


