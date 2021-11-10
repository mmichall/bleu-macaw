import os

import numpy as np
from nltk import tokenize
from collections import Counter, defaultdict

import torch
import nltk
from pandas import Series
from torch.utils.data import Dataset
from torchtext.vocab import Vocab, build_vocab_from_iterator

from readers.reader import Reader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

nltk.download('punkt')


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
        # self.ZEROS = np.zeros(self.vocab_len, dtype=np.int8)
        # self.diagonal = np.zeros((self.vocab_len, self.vocab_len), dtype=np.int8)
        # np.fill_diagonal(self.diagonal, 1)

    def __len__(self):
        return 100_000 #len(self.reader)

    def __getitem__(self, idx):
        item: Series = self.reader.read_example(idx)
        sentences = [SPECIALS.SOS + ' ' + sentence + ' ' + SPECIALS.EOS for sentence in tokenize.sent_tokenize(item['text'])]
        text = ' '.join(sentences)
        tokenized = self.tokenizer(text)
        _len = len(tokenized)
        if _len > self.sequence_length:
            tokenized = tokenized[:self.sequence_length]
        else:
            tokenized.extend(['<pad>'] * (self.sequence_length - _len))
        # 'stoi' instead of 'vocab' for optimalization purposes
        tokenized = [self.stoi[token] if token in self.stoi else 0 for token in tokenized]
        return idx, tokenized

    def build_vocab(self, min_freq: int = 1) -> Vocab:
        texts: Series = self.reader.read_column('text')
        texts = texts.apply(self.tokenizer)
        vocab: Vocab = build_vocab_from_iterator(texts,
                                                 min_freq=min_freq if min_freq else 1,
                                                 specials=[SPECIALS.UNK, SPECIALS.PAD, SPECIALS.SOS, SPECIALS.EOS])
        vocab.set_default_index(0)
        return vocab

    def ohe(self, idx):
        return self.diagonal[idx]


def build_vocab(dataset: Dataset, most_common_words: int, cache_uid: str, rebuild=False):
    vocab_path = f'.cache/vocab/_{cache_uid}_{most_common_words}.vb'
    if os.path.exists(vocab_path) and not rebuild:
        print(f'Read from cache {vocab_path}')
        vocab = torch.load(vocab_path)
    else:
        print(f'Building {cache_uid} vocabulary...')
        counter = Counter(dict(dataset.vocab.freqs.most_common(most_common_words)))
        vocab = Vocab(counter)
        stoi = defaultdict('<unk>')
        for word in vocab.stoi:
            stoi[word] = len(stoi)
        vocab.stoi = stoi
        torch.save(vocab, vocab_path)
        print(f'Saved in {vocab_path}')
    return vocab


class SPECIALS:
    UNK = '<unk>'
    PAD = '<pad>'
    SOS = '<sos>'
    EOS = '<eos>'


