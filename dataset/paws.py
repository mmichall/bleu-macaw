from abstract import ParaCorpus


class PAWSParaCorpus(ParaCorpus):

    def __init__(self, seed):
        super().__init__('paws', ['train'], ['sentence1', 'sentence2'], label_column='is_duplicate', seed=seed, configs='labeled_final')

    def read_para_pair(self, row):
        return (row['sentence1'], row['sentence2']), row['label']

