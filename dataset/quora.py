from abstract import ParaCorpus


class QuoraParaCorpus(ParaCorpus):

    def __init__(self, seed):
        super().__init__('quora', ['train'], ['questions'], label_column='is_duplicate', seed=seed)

    def read_para_pair(self, row):
        return row[self.text_columns[0]]['text'], row[self.label_column]

    def read_ids(self, row):
        return row[self.text_columns[0]]['id']

