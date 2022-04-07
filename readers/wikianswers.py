import os

from readers.reader import Reader, MODE


class WikiAnswersReader(Reader):

    def __init__(self, root, file_name='wikianswers/train.csv',
                 text_column_idx=1, mode: MODE = MODE.IN_MEMORY,
                 nrows=None, delimiter=',', randomly_to=None):

        super().__init__(os.path.join(root, file_name), text_column_idx, mode, nrows, delimiter, randomly_to=randomly_to)

