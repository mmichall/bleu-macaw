import math
import os
import random
import typing

import pandas as pd

from readers.reader import Reader, MODE

from nltk import tokenize

'''
torchtext.datasets.AmazonReviewPolarity
'''


class BooksReader(Reader):

    def __init__(self, root='E:\\books1\\epubtxt', file_name='', header=None,
                 text_column_idx=0, mode: MODE = MODE.IN_MEMORY, nrows=None, offset=200):
        super().__init__(os.path.join(root, file_name), text_column_idx, mode, nrows, header=header, offset=offset)

    def read(self) -> typing.Callable:
        return pd.read_fwf

    def preprocess(self, example):
        if not example or pd.isna(example):
            return ''
        try:
            sentences = tokenize.sent_tokenize(example)
        except:
            print(example)
        # get only first sentence
        if len(sentences) > 1:
            # because of cut last sentence
            sentences.pop()
        sentence = sentences[0].strip(' \'\"\t-')
        if not sentence:
            return ''
        return sentence
