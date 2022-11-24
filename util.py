from typing import List

import numpy as np
from sklearn.preprocessing import normalize


def filter_best(sentence: str, output_sent: List[str], encoder):
    original_embedding = normalize(encoder.encode([sentence]), axis=1)
    sentences = list(output_sent)
    embeddings = normalize(encoder.encode(sentences), axis=1)
    sim = np.matmul(original_embedding, np.transpose(embeddings))
    sim[sim > .95] = .0
    idx = np.argmax(sim)
    return sentences[idx]