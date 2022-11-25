import argparse

from msrp import MsrpParaCorpus
from quora import QuoraParaCorpus

import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        choices=['quora', 'msrp'],
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
    args = parser.parse_args()

    if args.dataset_name == 'quora':
        QuoraParaCorpus().generate(valid_size=args.valid_size, test_size=args.test_size)
    if args.dataset_name == 'msrp':
        MsrpParaCorpus().generate(valid_size=100, test_size=500)