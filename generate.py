import os
import torch
import argparse

from sentence_transformers import SentenceTransformer
from torchtext.vocab import Vocab
from transformers import AutoTokenizer

from model import RNNTextGeneratingModel

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def main(args):
    with open(args.data_dir + '/vocab/amazon_polarity_base_english_10.vb', 'r') as _:
        vocab: Vocab = torch.load(args.data_dir + '/vocab/amazon_polarity_base_english_10.vb')

    sentence_transformer = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    model = RNNTextGeneratingModel(sentence_transformer=sentence_transformer,
                                   rnn_size=256,
                                   rnn_dropout=0.,
                                   target_embedding_dim=256,
                                   target_embedding_dropout=0.,
                                   num_layers=1,
                                   bidirectional=False,
                                   vocab_size=len(vocab))

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)
    model.eval()

    words = []
    to_paraphrase = ['Lord of the Rings is the best book I\'ve ever read.']
    predictions = model.paraphrase(to_paraphrase=to_paraphrase)
    for word_index in predictions:
        x = vocab.get_itos()[word_index]
        words.append(x)

    print(f'Original: {to_paraphrase}')
    pretty = ''.join(words).replace('▁', ' ').replace('<pad>', '')
    print(f'Paraphrased: {pretty}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str,
                        default=os.path.join('E:', 'checkpoints', "gru_amazon_2_1.32713.pytorch"))
    parser.add_argument('-n', '--num_samples', type=int, default=8)

    parser.add_argument('-dd', '--data_dir', type=str, default='.cache')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=80)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='lstm')
    parser.add_argument('-hs', '--hidden_size', type=int, default=100)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.)
    parser.add_argument('-ls', '--latent_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', default=False)

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
