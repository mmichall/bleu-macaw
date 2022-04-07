import os
import torch
import argparse

from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from tokenizers.implementations import BertWordPieceTokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from transformers import DistilBertTokenizerFast, AutoTokenizer

import config
from dataset.pile import LanguageModelDataset
from model import RNNTextParaphrasingModel

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def main(args):
    # dataset: Dataset = load_dataset("cc_news", split="train", cache_dir=f'{config.data_path}')

    tokenizer: BertWordPieceTokenizer = BertWordPieceTokenizer.from_file(f'{config.root_path}/bert-tokenizer/vocab.txt')
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=32)
    tokenizer.enable_truncation(max_length=32)
    vocab = tokenizer.get_vocab()
    rvocab = {v: k for k, v in vocab.items()}
    # vocab_path = os.path.join(config.cache_path, 'vocab', 'books_5.vb')
    # with open(vocab_path, 'r') as _:
    #     vocab: Vocab = torch.load(vocab_path)

    sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L3-v2').to(device)
    dataset = LanguageModelDataset(path="bookcorpus", tokenizer=tokenizer, sentence_transformer=sentence_transformer)

    model = RNNTextParaphrasingModel(sentence_transformer=sentence_transformer,
                                     rnn_size=1024,
                                     rnn_dropout=0.3,
                                     embedding_dim=400,
                                     embedding_dropout_rate=0.,
                                     pad_index=tokenizer.padding['pad_id'],
                                     num_layers=2,
                                     bidirectional=False,
                                     vocab_size=tokenizer.get_vocab_size())

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)
    model.eval()

    # 1. epoch:
    # Original: ["This great little product fits well on my arm and doesn't slip."]
    # Paraphrased:  This item doesn't hold great little and well on my arm.</s>
    # Original: ['He is my favorite author and I love every book he has written.']
    # Paraphrased:  I love his books and his best book ever written.</s>

    for original in dataset.dataset['test'].shuffle().select(range(100)):
        words = []
        predictions = model.paraphrase([original['text']])
        for word_index in predictions:
            x = rvocab[word_index]
            words.append(x)

        print(f'Original: {original}\n')
        # pretty = ''.join(words).replace('▁', ' ').replace('<pad>', '')
        pretty = ' '.join(words).replace('[PAD]', '')
        print(f'Paraphrased: {pretty}\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str,
                        default=os.path.join('/praid/.cache', "model_dropout_0_0.9809.p"))
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
