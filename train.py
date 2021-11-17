import argparse

import torch
from sentence_transformers import SentenceTransformer
from torch import optim
from torch.nn import NLLLoss
from transformers import AutoTokenizer, BertTokenizer

import config
from dataset import LanguageModelingDataset, SPECIALS
from model import RNNTextGeneratingModel
from readers.amazon_polarity import AmazonPolarityReader
from readers.wykop import WykopReader
from trainer import RNNTextGeneratingModelTrainer

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def run(args):
    ## xlm-roberta-base
    # tokenizer = AutoTokenizer.from_pretrained(
    #     'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
    reader = WykopReader(nrows=320_000, root='E:\wykop')
    dataset = LanguageModelingDataset(reader=reader,
                                      tokenizer=tokenizer,
                                      min_freq=10,
                                      sequence_length=64,
                                      uuid='wykop_10',
                                      word_dropout=0.2)
    sentence_transformer = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    model = RNNTextGeneratingModel(sentence_transformer=sentence_transformer,
                                   rnn_size=256,
                                   rnn_dropout=0.,
                                   target_embedding_dim=256,
                                   target_embedding_dropout=0.,
                                   pad_index=dataset.vocab[SPECIALS.PAD],
                                   num_layers=1,
                                   bidirectional=False,
                                   vocab_size=dataset.vocab_len)

    optimizer = optim.Adam(params=model.parameters(), lr=0.0005)
    criterion = NLLLoss(reduction='mean')
    trainer = RNNTextGeneratingModelTrainer(dataset=dataset, model=model, optimizer=optimizer, criterion=criterion,
                                            epochs=100_000, batch_size=8)

    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--language_modeling_dataset', required=False, default='WikiText2')
    parser.add_argument('--tokenizer', required=False, default='basic_english')

    args = parser.parse_args()
    run(args)


