import os

from readers.reader import Reader, MODE

'''
torchtext.datasets.AmazonReviewPolarity
'''
class QuoraReader(Reader):

    def __init__(self, root, file_name='quora/questions.csv', header=0,
                 text_column_idx=3, mode: MODE = MODE.IN_MEMORY, nrows=None):
        super().__init__(os.path.join(root, file_name), text_column_idx, mode, nrows, header=header)
        # because of strange error raise
        self.df = self.df.drop(363416)
        self.df.reset_index(drop=True)

    def read_example(self, idx) -> str:
        if idx == 363416:
            return '<unk>'
        else:
            self.offset = idx
            return self.read_next()

