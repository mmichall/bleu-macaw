import json
import random
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

random.seed(42)


def split_train_test(ids, test_n=30_000):
    test_keys = random.sample(list(ids), test_n)
    train_keys = [k for k in tqdm(ids) if k not in test_keys]
    return train_keys, test_keys


quora = load_dataset('quora')['train']

res = defaultdict(lambda: [])
res_text = defaultdict(lambda: [])
id_sentence = {}
all = set()
for sentences_pair in tqdm(quora):
    text = sentences_pair['questions']['text']
    all.add(text[0])
    all.add(text[1])
    if sentences_pair['is_duplicate']:
        ids = sentences_pair['questions']['id']
        id_sentence[ids[0]] = text[0]
        id_sentence[ids[1]] = text[1]


with open("id_sentence.json", "w") as fp:
    json.dump(id_sentence, fp)

for sentences_pair in tqdm(quora):
    if sentences_pair['is_duplicate']:
        ids = sentences_pair['questions']['id']
        res[ids[0]].append(ids[1])

with open("ids.json", "w") as fp:
    json.dump(res, fp)


with open("ids.json", "r") as fp:
    ids = json.load(fp)

train_keys, valid_keys = split_train_test(ids, 1_000)
train_keys, test_keys = split_train_test(train_keys, 10_000)

with open("id_sentence.json", "r") as fp:
    id_sentence = json.load(fp)

with open("test.txt", "w", encoding='utf-8') as fp:
    for key in test_keys:
        print('\t'.join([id_sentence[str(key)], str([id_sentence[str(id)] for id in ids[key]])]), file=fp)
        try:
            all.remove(id_sentence[str(key)])
            for id in ids[key]:
                all.remove(id_sentence[str(id)])
        except:
            pass

with open("dev.txt", "w", encoding='utf-8') as fp:
    for key in valid_keys:
        for value_key in ids[key]:
            # print(''.join(['<quora>', id_sentence[str(key)], ' paraphrased: ', id_sentence[str(value_key)], '<|endoftext|>']), file=fp) # for supervised methods
            # print(''.join(['<quora>', id_sentence[str(key)], '<|endoftext|>']), file=fp)
            print(' '.join([id_sentence[str(key)], '>>>>', id_sentence[str(value_key)]]), file=fp)  # for supervised methods
            # print(id_sentence[str(key)], file=fp)
            try:
                all.remove(id_sentence[str(key)])
                for id in ids[key]:
                    all.remove(id_sentence[str(id)])
            except:
                pass

with open("train.txt", "w", encoding='utf-8') as fp:
    for key in train_keys:
        for value_key in ids[key]:
            # print(''.join(['<quora>', id_sentence[str(key)], ' paraphrased: ', id_sentence[str(value_key)], '<|endoftext|>']), file=fp)
            print(' '.join([id_sentence[str(key)], '>>>>', id_sentence[str(value_key)]]), file=fp)
            # print(id_sentence[str(key)], file=fp)

with open("task_adaptation.txt", "w", encoding='utf-8') as fp:
    for sentence in all:
        print(''.join(['<quora>', sentence, '<|endoftext|>']), file=fp)