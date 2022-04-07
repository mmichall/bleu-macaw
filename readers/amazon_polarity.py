import os

from readers.reader import Reader, MODE
from nltk import tokenize

'''
torchtext.datasets.AmazonReviewPolarity
'''
class AmazonPolarityReader(Reader):

    def __init__(self, root, file_name='amazon_review_polarity_csv/train.csv',
                 text_column_idx=2, mode: MODE = MODE.IN_MEMORY, nrows=None):
        super().__init__(os.path.join(root, file_name), text_column_idx, mode, nrows)

    def preprocess(self, example):
        sentences = tokenize.sent_tokenize(example)
        # get only first sentence
        return sentences[0]
