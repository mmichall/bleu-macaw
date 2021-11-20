import os

import numpy as np
from nltk import tokenize

import torch
import nltk
from pandas import Series
from torch.utils.data import Dataset
from torchtext.vocab import Vocab, build_vocab_from_iterator

import config
from readers.reader import Reader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

nltk.download('punkt')


class LanguageModelingDataset(Dataset):
    def __init__(self, reader: Reader, vocab: Vocab = None, tokenizer=None, min_freq=None, sequence_length=100,
                 uuid=None, word_dropout=.0, transform_to_tensor=True, reset_vocab=False):
        self.reader = reader
        self.vocab: Vocab = vocab
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.transform_to_tensor = transform_to_tensor
        self.word_dropout = word_dropout
        self.reset_vocab = reset_vocab
        if not self.vocab or self.reset_vocab:
            path = f'{config.cache_path}/vocab/{uuid}_{min_freq}.vb'
            if os.path.exists(path) and not self.reset_vocab:
                self.vocab = torch.load(path)
                print(f'Vocabulary has been loaded from {path}')
            else:
                print('Vocabulary is building...', end='')
                self.vocab = self.build_vocab(min_freq=min_freq)
                torch.save(self.vocab, path)
                print(f'DONE. Cached in {path}')
        self.vocab_len = len(self.vocab)
        self.stoi = self.vocab.get_stoi()

    def __len__(self):
        return len(self.reader)

    # TODO: optimalization + cache for tokenized dataset ASAP
    def __getitem__(self, idx):
        example: str = self.reader.read_example(idx)
        raw = example
        tokenized = self.tokenizer.tokenize(example)
        _len = len(tokenized)
        if _len > self.sequence_length:
            _len = self.sequence_length - 1
            tokenized = tokenized[:self.sequence_length]
        elif _len == self.sequence_length:
            _len = self.sequence_length - 1
        else:
            tokenized.extend([SPECIALS.PAD] * (self.sequence_length - _len))
        # 'stoi' instead of 'vocab' for optimization purposes
        tokenized = np.array([self.stoi[token] if token in self.stoi else 0 for token in tokenized])
        target = tokenized.copy()
        if self.word_dropout > 0:
            # randomly replace decoder input with <mask>
            prob = np.random.rand(self.sequence_length)
            prob[(tokenized - self.vocab[SPECIALS.PAD]) == 0] = 1
            tokenized[prob < self.word_dropout] = self.stoi[SPECIALS.MSK]
        tokenized = np.insert(tokenized[:-1], 0, [self.stoi[SPECIALS.SOS]], axis=0)
        target[_len] = self.stoi[SPECIALS.EOS]
        tokenized[_len if _len == self.sequence_length-1 else _len+1] = self.stoi[SPECIALS.EOS]
        if self.transform_to_tensor:
            tokenized = torch.tensor(tokenized, device=device, dtype=torch.int).T
            target = torch.tensor(target, device=device, dtype=torch.long).T
        return idx, raw, tokenized, target

    def build_vocab(self, min_freq: int = 1) -> Vocab:
        texts: Series = self.reader.read_text_column()
        texts = texts.apply(lambda example: self.tokenizer.tokenize(tokenize.sent_tokenize(example)[0])[:self.sequence_length])
        vocab: Vocab = build_vocab_from_iterator(texts,
                                                 min_freq=min_freq if min_freq else 1,
                                                 specials=[SPECIALS.UNK, SPECIALS.PAD, SPECIALS.SOS, SPECIALS.EOS, SPECIALS.MSK])
        vocab.set_default_index(0)
        return vocab


class SPECIALS:
    UNK = '<unk>'
    PAD = '<pad>'
    SOS = '<s>'
    EOS = '</s>'
    MSK = '<mask>'


