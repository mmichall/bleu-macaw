from ast import literal_eval

import tqdm
from transformers import pipeline

#1 lang  'fr'
#2 lang = 'de'
#3 lang = 'ru'
#4 lang = 'es'
#5 lang = 'zh'
#6 lang = 'fi'
#7 lang = 'it'
#8 lang = 'zls'
#9 lang = 'hu'
lang = 'bg'

encoder_name = "paraphrase-multilingual-mpnet-base-v2"
first_model_name = f'Helsinki-NLP/opus-mt-en-{lang}'
translator_first = pipeline("translation", model=first_model_name, device=0)
second_model_name = f'Helsinki-NLP/opus-mt-{lang}-en'
translator_second = pipeline("translation", model=second_model_name, device=0)


def format_batch_texts(language_code, batch_texts):
    formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]
    return formated_bach

def perform_translation(batch_texts, pipe, language="fr"):
    formated_batch_texts = format_batch_texts(language, batch_texts)
    return pipe(formated_batch_texts, max_length=1024)[0]['translation_text']

def perform_back_translation(batch_texts, original_language="en", temporary_language="fr"):

  tmp_translated_batch = perform_translation(batch_texts, translator_first, temporary_language)
  back_translated_batch = perform_translation([tmp_translated_batch], translator_second, original_language)
  return back_translated_batch


if __name__ == '__main__':
    with open('../.data/quora/test.tsv', "r", encoding='utf-8') as f:
        lines = [line.rstrip() for line in f]
    with open(f'../results/back_transl/{lang}.tsv', "w", encoding='utf-8') as f:
        for line in tqdm.tqdm(lines):
            input, refs = line.split('\t')
            orginal_input = input
            back_translated_batch = perform_back_translation([orginal_input])
            print(back_translated_batch, file=f)
            # print('\t'.join([orginal_input, str(the_best), str(cands), str(refs)]), file=f)
