import os
import shutil

for file in os.listdir("/praid/.cache/"):
    if file.startswith("model_dropout"):
        print(file)
        # os.remove('')

# shutil.rmtree('/praid/.cache/sentence-transformers_paraphrase-multilingual-mpnet-base-v2')
# shutil.rmtree('/praid/.cache/gpt2-paraphrase-multilingual-mpnet-base-v2-english-amazon')
# shutil.rmtree('/praid/.cache/gpt2-paraphrase-multilingual-mpnet-base-v2-english-glue_qqp')
# shutil.rmtree('/praid/.cache/gpt2-paraphrase-multilingual-mpnet-base-v2-english-glue_qqp_lower')
# shutil.rmtree('/praid/.cache/sentence-transformers_paraphrase-MiniLM-L3-v2')
# shutil.rmtree('/praid/.cache/gpt2-paraphrase-MiniLM-L3-v2-english-glue_qqp_lower')
# shutil.rmtree('/praid/.cache/sentence-transformers_paraphrase-albert-small-v2')
# shutil.rmtree('/praid/.cache/gpt2-paraphrase-albert-small-v2-english-glue_qqp')
# shutil.rmtree('/praid/.cache/sentence-transformers_paraphrase-mpnet-base-v2')
# shutil.rmtree('/praid/.cache/gpt2-paraphrase-mpnet-base-v2-english-glue_qqp')
# shutil.rmtree('/praid/.cache/gpt2-paraphrase-mpnet-base-v2-english-quora_1e')
# shutil.rmtree('/praid/.cache/gpt2-paraphrase-mpnet-base-v2-english-quora_10e')
# shutil.rmtree('/praid/.cache/gpt2-paraphrase-multilingual-mpnet-base-v2-english-quora_mmpnet_10e')
# shutil.rmtree('/praid/.cache/gpt2-paraphrase-multilingual-mpnet-base-v2-english-quora_mmpnet_1e')