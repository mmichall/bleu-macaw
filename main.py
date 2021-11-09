import argparse
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer

from dataset import LanguageModelingDataset
from readers.amazon_polarity import AmazonPolarityReader


def run(args):
    tokenizer = get_tokenizer('basic_english')
    dataset = LanguageModelingDataset(reader=AmazonPolarityReader(),
                                      tokenizer=tokenizer,
                                      min_freq=20,
                                      sequence_length=100,
                                      uuid='amazon_polarity_base_english')
    dl = DataLoader(dataset, batch_size=128, shuffle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--language_modeling_dataset', required=False, default='WikiText2')
    parser.add_argument('--tokenizer', required=False, default='basic_english')

    args = parser.parse_args()
    run(args)


