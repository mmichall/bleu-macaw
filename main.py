import argparse
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from tqdm import tqdm

from dataset import LanguageModelingDataset, SPECIALS
from model import SentenceVAE
from readers.amazon_polarity import AmazonPolarityReader
from train import LanguageModelingTrainer


def run(args):
    tokenizer = get_tokenizer('basic_english')
    dataset = LanguageModelingDataset(reader=AmazonPolarityReader(),
                                      tokenizer=tokenizer,
                                      min_freq=10,
                                      sequence_length=80,
                                      uuid='amazon_polarity_base_english')

    model = SentenceVAE(vocab_size=len(dataset.vocab),
                        embedding_size=784,
                        rnn_type='gru',
                        hidden_size=600,
                        word_dropout=0.5,
                        embedding_dropout=0.5,
                        latent_size=1200,
                        sos_idx=dataset.vocab[SPECIALS.SOS],
                        eos_idx=dataset.vocab[SPECIALS.EOS],
                        pad_idx=dataset.vocab[SPECIALS.PAD],
                        unk_idx=dataset.vocab[SPECIALS.UNK],
                        max_sequence_length=80,
                        num_layers=1,
                        bidirectional=False)

    trainer = LanguageModelingTrainer(dataset, model, epochs=20, batch_size=16, shuffle=True)
    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--language_modeling_dataset', required=False, default='WikiText2')
    parser.add_argument('--tokenizer', required=False, default='basic_english')

    args = parser.parse_args()
    run(args)


