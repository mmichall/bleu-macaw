import argparse
import os
import random
import shutil

import torch
from sentence_transformers import SentenceTransformer
from torch import optim
from torch.nn import NLLLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer

from dataset.pile import LanguageModelDataset
from model import RNNTextParaphrasingModel
from trainer import ParaphrasingModelTrainer

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from torch.multiprocessing import set_start_method
set_start_method("spawn")


def run(args):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    tokenizer.enable_padding(length=32)
    tokenizer.enable_truncation(max_length=32)

    sentence_transformer = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2').to(device)
    # sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L3-v2').to(device)
    dataset = LanguageModelDataset(path="bookcorpus", tokenizer=tokenizer, sentence_transformer=sentence_transformer)
    dataset.build()
    model = RNNTextParaphrasingModel(sentence_transformer=sentence_transformer,
                                     rnn_size=1024,
                                     rnn_dropout=0.3,
                                     embedding_dim=400,
                                     embedding_dropout_rate=0.,
                                     pad_index=tokenizer.padding['pad_id'],
                                     num_layers=2,
                                     bidirectional=False,
                                     vocab_size=tokenizer.get_vocab_size())

    optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
    criterion = NLLLoss(reduction='mean')
    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3, threshold=1e-5)
    trainer = ParaphrasingModelTrainer(dataset=dataset, model=model, optimizer=optimizer, criterion=criterion,
                                       scheduler=scheduler, epochs=100_000, batch_size=128)

    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--language_modeling_dataset', required=False, default='WikiText2')
    parser.add_argument('--tokenizer', required=False, default='basic_english')

    args = parser.parse_args()
    run(args)
