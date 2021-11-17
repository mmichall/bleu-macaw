import os

from nltk import tokenize

from readers.reader import Reader, MODE


class WykopReader(Reader):

    def __init__(self, root, file_name='entries.txt',
                 text_column_name=None, mode: MODE = MODE.IN_MEMORY,
                 nrows=None, delimiter='\t'):

        super().__init__(os.path.join(root, file_name), text_column_name, mode, nrows, delimiter)

    def read_next(self):
        example: str = super().read_next()
        sentences = tokenize.sent_tokenize(example)
        # get only first sentence
        return sentences[0]