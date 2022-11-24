import numpy as np
from numpy import mean

output_file = '../stanford-corenlp-4.5.1/output_t5_4.txt'


scores = []
with open(output_file, 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('# Parse 1 with score'):
            scores.append(float(line.split(' ')[-1]))

print(mean(np.array(scores)))