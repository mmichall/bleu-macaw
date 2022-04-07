from pandas import DataFrame

import config
from readers.wikianswers import WikiAnswersReader

if __name__ == '__main__':

    reader = WikiAnswersReader(root=f'../{config.data_path}')
    train: DataFrame = reader.df.sample(n=10_000_000)
    valid = reader.df.sample(n=10_000)
    test = reader.df.sample(n=40_000)

    train.to_csv('../.data/wikianswers/train.csv', header=False, index=False)
    valid.to_csv('../.data/wikianswers/valid.csv', header=False, index=False)
    test.to_csv('../.data/wikianswers/test.csv', header=False, index=False)
