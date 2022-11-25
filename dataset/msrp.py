from abstract import ParaCorpus


class MsrpParaCorpus(ParaCorpus):

    def __init__(self):
        super().__init__('HHousen/msrp', 'train', ['sentence1', 'sentence2'], label_column='label')

    def read_para_pair(self, row):
        return (row[self.text_columns[0]], row[self.text_columns[1]]), row[self.label_column]


