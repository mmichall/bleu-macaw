import random
import string
from typing import List

import datasets
import evaluate
import numpy as np
import sentence_transformers
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import normalize

encoder = SentenceTransformer("sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1").cuda()

bleu_metric = datasets.load_metric('bleu', max_order=2)
def filter_best(sentence: str, output_sent: List[str]):

    # bleu_input_cands = bleu_metric.compute(predictions=[sentence.split()], references=[[sent.split() for sent in output_sent]], max_order=4)
    agg = []
    if len(list(output_sent)) == 0:
        output_sent.append('dummy one ###')
    cos_sim = util.cos_sim(encoder.encode(sentence, normalize_embeddings=True, convert_to_tensor=True),
                           encoder.encode(list(output_sent), normalize_embeddings=True, convert_to_tensor=True))
    sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower()
    for gen in output_sent:
        gen = gen.translate(str.maketrans('', '', string.punctuation))
        try:
            results = bleu_metric.compute(predictions=[sentence.split()], references=[[gen.lower().split()]], max_order=2)
            agg.append(results['bleu'])
        except:
            agg.append(1)

    beta = 2.0
    epsilon = 0.0000001
    res = []
    for bleu, sbert_sim in zip(agg, cos_sim[0]):
        res.append(pow((beta * pow(sbert_sim.cpu().numpy(), -1) + pow((1. + epsilon) - bleu, -1)) / (beta + 1.0), -1))

    sentences = list(output_sent)
    # # lev_sim = np.array(normalized_damerau_levenshtein_distance_seqs(sentence, sentences))
    # # lev_sim[lev_sim < 0.1] = 0
    # # lev_sim[lev_sim >= 0.1] = 1.0
    # lev_sim = [0 if _sen.lower() == sentence.lower() else 1 for _sen in sentences]
    # original_embedding = encoder.encode(sentence, convert_to_tensor=True)
    # embeddings = encoder.encode(sentences, convert_to_tensor=True)
    # sim = util.cos_sim(original_embedding, embeddings)
    # sim = lev_sim * sim.cpu().numpy()
    # sim = sim[sim <= 0.99]
    idx = np.argmax(res)
    return sentences[idx]


# prediction = "hello there general"
# references = ["hello there general kenobi", "hello there general xioami"]
#
# bleu = datasets.load_metric("bleu")
# agg = []
# for ref in references:
#     results = bleu.compute(predictions=[prediction.split()], references=[[ref.split()]], max_order=2)
#     agg.append(results['bleu'])
# print(np.mean(agg))
