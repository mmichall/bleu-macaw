import ast

result_file_path = '../results/t5_beam_search.txt'
output_file = 't5_beam_search_grammar.txt'


with open(result_file_path, 'r', encoding='utf8') as f, open(output_file, 'w', encoding='utf8') as out:
    lines = f.readlines()
    for line in lines:
        cands = ast.literal_eval(line.split('\t')[2])
        for cand in cands:
            if len(cand) > 100:
                cand = cand[:100]
            out.write(cand + '\n')