import os
import typing

import pandas as pd
from readers.reader import Reader, MODE


class BeletrystykaReader(Reader):

    def __init__(self, root, file_name='beletrystyka.out',
                 text_column_name=None, mode: MODE = MODE.IN_MEMORY,
                 nrows=None, delimiter=None):

        super().__init__(os.path.join(root, file_name), text_column_name, mode, nrows, delimiter)

    def read(self) -> typing.Callable:
        return pd.read_fwf
