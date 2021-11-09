from abc import ABC, abstractmethod
from enum import Enum


class MODE(Enum):
    IN_MEMORY = 1
    LAZY = 2


class Reader(ABC):

    def __init__(self, root='.data', tokenizer=None):
        self.root = root
        self.tokenizer = tokenizer

    @abstractmethod
    def read_example(self, idx):
        return NotImplemented

    @abstractmethod
    def read_next(self):
        return NotImplemented

    @abstractmethod
    def read_column(self, name, tokenizer):
        return NotImplemented

    def __len__(self):
        return self.len



