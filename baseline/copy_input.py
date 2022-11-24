from ast import literal_eval

import tqdm


class NaiveParaphraseGenerator:

    def generate(self, input):
        return [input] * 5, input


generator: NaiveParaphraseGenerator = NaiveParaphraseGenerator()
with open('../.data/quora/test.txt', "r", encoding='utf-8') as f:
    lines = [line.rstrip() for line in f]
with open('results/copy_input.txt', "w", encoding='utf-8') as f:
    for line in tqdm.tqdm(lines):
        input, refs = line.split('\t')
        orginal_input = input
        refs = literal_eval(refs)
        refs = [ref.lower() for ref in refs]
        cands, the_best = generator.generate(input)
        print(input, the_best)
        print('\t'.join([orginal_input, str(the_best), str(cands), str(refs)]), file=f)