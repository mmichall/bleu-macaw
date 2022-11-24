from ast import literal_eval

from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


model = T5ForConditionalGeneration.from_pretrained('prithivida/parrot_paraphraser_on_T5', cache_dir="PATH/TO/MY/CACHE/DIR")
tokenizer = T5Tokenizer.from_pretrained('prithivida/parrot_paraphraser_on_T5', cache_dir="PATH/TO/MY/CACHE/DIR")


with open('.data/quora/test.txt', "r", encoding='utf-8') as f:
    lines = [line.rstrip() for line in f]


def _process_text(text: str):
    text = text.replace("<br />", " ")
    # paraphrased: <quora>
    # text = ''.join(['<quora>', text, '<|endoftext|>']) # quora -para
    # text = ''.join(['<quora>', text, ' paraphrased: ']) # quora -para
    text = ' '.join([text, '<|endoftext|>'])
    return text

with open('results/t5_beam_search.txt', "w", encoding='utf-8') as f:
    for line in tqdm(lines):
        input, refs = line.split('\t')
        orginal_input = input
        refs = literal_eval(refs)
        # refs = [ref.lower() for ref in refs]
        input = _process_text(input)
        tokenized = tokenizer(input, return_tensors='pt')

        generated_ids = model.generate(tokenized['input_ids'],
                                       num_beams=10,
                                       num_return_sequences=5,
                                       temperature=2,
                                       num_beam_groups=5,
                                       diversity_penalty=2.0,
                                       no_repeat_ngram_size=2,
                                       early_stopping=True,
                                       length_penalty=2)
        cands = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        print('\t'.join([orginal_input, '###', str(cands), str(refs)]), file=f)
# generated_ids = model.generate(batch['input_ids'],
#                                 num_beams=5,
#                                 num_return_sequences=5,
#                                 temperature=1.5,
#                                 num_beam_groups=5,
#                                 diversity_penalty=2.0,
#                                 no_repeat_ngram_size=2,
#                                 early_stopping=True,
#                                 length_penalty=2.0)
#
# generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)