import os

from readers.reader import Reader, MODE
from nltk import tokenize

'''
torchtext.datasets.AmazonReviewPolarity
'''
class AmazonPolarityReader(Reader):

    def __init__(self, root, file_name='amazon_review_polarity_csv',
                 text_column_name='text', mode: MODE = MODE.IN_MEMORY,
                 nrows=None):
        super().__init__(os.path.join(root, file_name), text_column_name, mode, nrows)

    def preprocess(self, example):
        sentences = tokenize.sent_tokenize(example)
        # get only first sentence
        return sentences[0]