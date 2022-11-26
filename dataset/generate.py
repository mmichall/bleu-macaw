import argparse

from paws import PAWSParaCorpus
from mscoco import MSCOCOParaCorpus
from msrp import MSRPParaCorpus
from quora import QuoraParaCorpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        choices=['quora', 'msrp', 'mscoco', 'paws'],
        required=True,
        help="dataset name. 'quora' available"
    )
    parser.add_argument(
        "--valid_size",
        type=int,
        required=False,
        help="Valid split size",
        default=1_000
    )
    parser.add_argument(
        "--test_size",
        type=int,
        required=False,
        help="Test split size",
        default=10_000
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Seed",
        default=42
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        help="Output dir. Default <root>/.data/<dataset_name>",
    )
    args = parser.parse_args()

    if args.dataset_name == 'quora':
        QuoraParaCorpus(args.seed).generate(valid_size=args.valid_size, test_size=args.test_size)
    if args.dataset_name == 'msrp':
        MSRPParaCorpus(args.seed).generate(valid_size=args.valid_size, test_size=args.test_size)
    if args.dataset_name == 'mscoco':
        MSCOCOParaCorpus(args.seed).generate(valid_size=args.valid_size, test_size=args.test_size)
    if args.dataset_name == 'paws':
        PAWSParaCorpus(args.seed).generate(valid_size=args.valid_size, test_size=args.test_size)