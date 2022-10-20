import os

from sentence_transformers import SentenceTransformer
from torch.nn import CosineSimilarity

import config

# model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
#
# from datasets import load_dataset
# dataset = load_dataset('quora', split='train')

# sentences = ['I don\'t love this music',
#     'I love this music']
# sentence_embeddings = model.encode(sentences, convert_to_numpy=False)
#
# cos = CosineSimilarity(dim=-1, eps=1e-6)
# print(cos(sentence_embeddings[0], sentence_embeddings[1]))
#
# print(sentence_embeddings[0].size())

# tokenizer = AutoTokenizer.from_pretrained(
#     'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
# tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
# reader = BooksReader(nrows=20_000_000)
# dataset = LanguageModelingDataset(reader=reader,
#                                   tokenizer=tokenizer,
#                                   min_freq=5,
#                                   sequence_length=64,
#                                   uuid='books',
#                                   word_dropout=0.,
#                                   to_lower=True)


import os, time

path = config.checkpoint_path
now = time.time()

for filename in os.listdir(path):
    # if os.stat(os.path.join(path, filename)).st_mtime < now - 7 * 86400:
    if os.path.isfile(os.path.join(path, filename)):
        print(filename)
        os.remove(os.path.join(path, filename))