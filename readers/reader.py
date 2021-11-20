from abc import ABC, abstractmethod
from enum import Enum
import typing

import pandas as pd
from pandas import Series, DataFrame


class MODE(Enum):
    IN_MEMORY = 1


class Reader(ABC):

    def __init__(self, root: str, text_column_name: str, mode: MODE = MODE.IN_MEMORY, nrows=None, delimiter=None):
        self.root = root
        self.mode = mode
        self.offset = 0
        self.nrows = nrows
        self.delimiter = delimiter
        self.text_column_name = text_column_name
        self.offset = 0
        if self.mode == MODE.IN_MEMORY:
            print(f'Reading {root} file...', end='')
            self.df: DataFrame = self.read()(f'{self.root}', nrows=nrows, delimiter=self.delimiter,
                                             skip_blank_lines=True,
                                             usecols=[self.text_column_name] if self.text_column_name else None)
            self.len = len(self.df)
            print(f'DONE. {self.len} examples have been read.')

    def read_example(self, idx) -> str:
        self.offset = idx
        return self.read_next()

    def read_next(self) -> str:
        if self.mode == MODE.IN_MEMORY:
            item = self.df.loc[self.offset]
            self.offset = self.offset + 1
            text = item[self.text_column_name] if self.text_column_name else item[0]
            return text

    def read_text_column(self) -> Series:
        self.reset()
        if self.mode == MODE.IN_MEMORY:
            return self.df[self.text_column_name] if self.text_column_name else self.df.iloc[:, 0]

    def reset(self):
        self.offset = 0

    def read(self) -> typing.Callable:
        return pd.read_csv

    def __len__(self):
        return self.len



