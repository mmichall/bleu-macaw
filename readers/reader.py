from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, List, Callable
import os

import pandas as pd
from pandas import Series, DataFrame
from sklearn.utils import shuffle


class MODE(Enum):
    IN_MEMORY = 1


class Reader(ABC):

    def __init__(self, root: str, text_column_idx: Union[int, List[int]], mode: MODE = MODE.IN_MEMORY,
                 nrows=None, delimiter=None, header=None, randomly_to=None, offset=0):
        self.root = root
        self.mode = mode
        self.offset = offset
        self.nrows = nrows
        self.delimiter = delimiter
        self.text_column_idx = text_column_idx
        self.len = 0
        self.header = header
        self.randomly_to = randomly_to
        if self.mode == MODE.IN_MEMORY:
            if os.path.isdir(self.root):
                for f in shuffle(os.listdir(self.root))[:750]:
                    print(f'Reading {os.path.join(self.root, f)} file...')
                    df_tmp: DataFrame = self.read()(os.path.join(self.root, f), nrows=randomly_to if randomly_to else nrows / 100,
                                                         delimiter=self.delimiter, skip_blank_lines=True,
                                                         header=self.header)[0]
                    df_tmp = df_tmp.to_frame()
                    if hasattr(self, 'df'):
                        self.df = self.df.append(df_tmp, ignore_index=True)
                    else:
                        self.df = df_tmp
                    self.len = len(self.df)
                print(f'DONE. {self.len} examples have been read.')
            else:
                print(f'Reading {root} file...', end='')
                self.df: DataFrame = self.read()(f'{self.root}', nrows=randomly_to if randomly_to else nrows,
                                                 delimiter=self.delimiter, skip_blank_lines=True, header=self.header)
                self.len = randomly_to if randomly_to else len(self.df)
                print(f'DONE. {self.len} examples have been read.')

    def read_example(self, idx) -> str:
        self.offset = idx
        return self.read_next()

    def read_next(self) -> str:
        if self.mode == MODE.IN_MEMORY:
            item = self.df.loc[self.offset]
            self.offset = self.offset + 1
            text = item[self.text_column_idx] if self.text_column_idx else item[0]
            return self.preprocess(text)

    def read_text_column(self) -> Series:
        self.reset()
        if self.mode == MODE.IN_MEMORY:
            return self.df.iloc[:, self.text_column_idx] if self.text_column_idx else self.df.iloc[:, 0]

    def reset(self):
        self.offset = 0

    def read(self) -> Callable:
        return pd.read_csv

    def preprocess(self, example):
        return example

    def __len__(self):
        return self.len



