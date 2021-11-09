import pandas as pd
from pandas import DataFrame, Series

from readers.reader import Reader, MODE


class AmazonPolarityReader(Reader):

    def __init__(self, root='.data/amazon_review_polarity_csv', mode: MODE = MODE.IN_MEMORY, tokenizer=None):
        super().__init__(root)
        self.mode = mode
        self.offset = 0
        self.tokenizer = tokenizer
        if mode == MODE.IN_MEMORY:
            print(f'Reading {root} file...', end='')
            self.df: DataFrame = pd.read_csv(f'{self.root}/train.csv', names=["polarity", "title", "text"])
            self.len = len(self.df)
            print('DONE')
            print(f'{self.len} examples have been read')

    def read_example(self, idx):
        self.offset = idx
        return self.read_next()

    def read_next(self):
        if self.mode == MODE.IN_MEMORY:
            item = self.df.loc[self.offset]
            self.offset = self.offset + 1
            return item

    def read_column(self, name: str, reset=True) -> Series:
        if reset:
            self.reset()
        if self.mode == MODE.IN_MEMORY:
            return self.df[name]

    def reset(self):
        self.offset = 0
