from abstract import ParaCorpus

"""
Microsoft Research Paraphrase Corpus (MRPC) is a corpus consists of 5,801 sentence pairs
collected from newswire articles. Each pair is labelled if it is a paraphrase or not by human annotators.
The whole set is divided into a training subset (4,076 sentence pairs of which 2,753 are paraphrases) and
a test subset (1,725 pairs of which 1,147 are paraphrases).

URL: https://paperswithcode.com/dataset/mrpc
"""
class MSRPParaCorpus(ParaCorpus):

    def __init__(self, seed):
        super().__init__('HHousen/msrp', ['train', 'test'], ['sentence1', 'sentence2'], label_column='label', seed=seed)

    def read_para_pair(self, row):
        return (row[self.text_columns[0]], row[self.text_columns[1]]), row[self.label_column]


