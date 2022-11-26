# BleuMacaw: GPT-2 and SentenceTransformers for Paraphrases Generation


<img src="https://ih1.redbubble.net/image.790396839.3293/st,small,845x845-pad,1000x1000,f8f8f8.u2.jpg" width="260" align="right" alt="Graphical Model diagram for HRQ-VAE" />



[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-sketch-induction-for-paraphrase/paraphrase-generation-on-mscoco)](https://paperswithcode.com/sota/paraphrase-generation-on-mscoco?p=hierarchical-sketch-induction-for-paraphrase)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-sketch-induction-for-paraphrase/paraphrase-generation-on-paralex)](https://paperswithcode.com/sota/paraphrase-generation-on-paralex?p=hierarchical-sketch-induction-for-paraphrase)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-sketch-induction-for-paraphrase/paraphrase-generation-on-quora-question-pairs-1)](https://paperswithcode.com/sota/paraphrase-generation-on-quora-question-pairs-1?p=hierarchical-sketch-induction-for-paraphrase)



This repo contains the code for the paper [BleuMacaw: Unsupervised Paraphrasing with Pre-trained Language Model and Sentence Transformers](...), by Michał Perełkiewicz, Sławomir Dadas & Rafał Poświata (ACL 2023/ EMNLP 2023).

We propose a auto-generative model of paraphrase generation based on the pre-trained OpenAI's GPT-2 model and the SentenceTransformers encoding.


## Table of contents
- [Installation](#installation)
- [Datasets preparation](#datasets_preparation)

## Installation

First, create a fresh virtualenv and install the dependencies:
```
pip install -r requirements.txt
```

You can also use the `./docker/Dockerfile` file to build the docker image and run code in a container.

## Datasets preparation

Then download (or create) the datasets/checkpoints you want to work with:

<a href="https://..." download>Download our split of QQP</a>

<a href="https://..." download>Download our split of MSRP</a>

<a href="https://..." download>Download our split of MSCOCO</a>

<a href="https://..." download>Download our split of Paws-Wiki</a>

Data zip should be should be unzipped into `./data/<dataset_name>`, eg `./models/quora`.

To generate data files you can also use the generation script:

```
python3 ./dataset/generate.py --dataset_name quora --valid_size 1_000 --test_size 10_000 --seed 42 --output_dir <path>
```

These should be folders `<dataset_name>` in `./data`, containing `{train,dev,test}.tsv` and `unsupervised.txt` files.
Files: `{train,dev,test}.tsv` are used while supervised method training (used only for reproduce competitive methods results) and `unsupervised.txt` for BLeuMacaw fine-tuning.
The default output dir is `<root>/.data/<dataset_name>`. Available datasets names: quora, msrp, paws (only Wiki), mscoco. Script automatically download the raw data files from huggingface hub and prepare splits fo training, validation, testing and for unsupervised manner learning in proper format.

### Files format
TODO
### Training on a new dataset
TODO
### A test dataset to use for evaluation
TODO
## Train the model

### Pre-training on large text corpus 

For adapt a BleuMacaw model to sentence data type, we pre-trained the model on the OpenWebText corpus (https://huggingface.co/datasets/openwebtext). 
The OpenWebText corpus is an open-source recreation of the WebText corpus (orginally used to train a GPT2 model).
The text is web content extracted from URLs shared on Reddit with at least three upvotes (40GB).

For optimization purpose, we limit the number of articles from the corpus to 1 000 000 (approx. 38 261 000 sentences).

To pre-train a BleuMacaw model please use the command:

```
nvidia-docker run --runtime=nvidia -v <project_root_path>:/proot -v <data_path>:/praid --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -d --gpus device=<n_gpu> -p port:port paraphrases python3 /proot/run_clm.py
```

This command:
<ol>
  <li>Download the OpenWebText data files and extract them in cache folder (about 40GB of text data)</li>
  <li>Download GPT-2 and SentenceTransformers models</li>
  <li>Tokenize the text data and cache them in cache folder (for 768 length vectors and 1M articles is approx 120GB of data).
The cached files are used for next training script run when te same SentenceEncoder and data articles limit is used.</li>
  <li>Start training the model and save checkpoint</li>
</ol>

### Fine-tuning
unsupervised.txt

command:

### Choosing GPT-2 and SentenceTransfomers models

#### GPT-2
**HuggingFace**'s GPT-2 implementation is available in five different sizes (https://huggingface.co/transformers/v2.2.0/pretrained_models.html): small (117M), medium (345M), large (774M), xl (1558M) and a distilled version of the small checkpoint: distilgpt-2 (82M).
For optimization purpose, we use a small GPT-2 model. The further research should include biggest bersions of the model.

#### SentenceTransformers
<ul>
  <li>parahrase-mpnet-base-v2</li>
  <li>all-mpnet-base-v2</li>
  <li>parahrase-distilroberta-base-v2</li>
  <li>multi-qa-distilbert-dot-v1 (0.2M articles)</li>
</ul>

## Evaluation
### Metrics
BLEU, selfBLEU, oriBLEU, ROGUE-{1,2,L}, Meteor, s-t sim-{input-generated, best-refs}, BERTScore, stanford-grammar, 

TODO: <miara ratio podobieństwo generated/ sim-sem generated-input>  

## Citation
