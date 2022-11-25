from typing import Set, Callable
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, TextGenerationPipeline
from transformers.pipelines.text_generation import ReturnType

from modified_gpt2 import GPT2ParaphrasingLM, DatasetSentenceSplitter


class ParaphrasingPipeline(TextGenerationPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, prompt_text, prefix=""):
        inputs = self.tokenizer(
            prefix + prompt_text, padding=True, add_special_tokens=True, return_tensors=self.framework, max_length=512,
            truncation=True
        )
        inputs["prompt_text"] = prompt_text
        return inputs

    def _sanitize_parameters(
            self,
            return_full_text=None,
            return_tensors=None,
            return_text=None,
            return_type=None,
            clean_up_tokenization_spaces=None,
            prefix='',
            **generate_kwargs
    ):
        preprocess_params = {}
        if prefix is not None:
            preprocess_params["prefix"] = prefix
        if prefix:
            prefix_inputs = self.tokenizer(
                prefix, padding=False, return_tensors=self.framework
            )
            prefix_length = prefix_inputs["input_ids"].shape[-1]
            if "max_length" in generate_kwargs:
                generate_kwargs["max_length"] += prefix_length
            else:
                generate_kwargs["max_length"] = self.model.config.max_length + prefix_length

            if "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        forward_params = generate_kwargs

        postprocess_params = {}
        if return_full_text is not None and return_type is None:
            return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
        if return_tensors is not None and return_type is None:
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        return preprocess_params, forward_params, postprocess_params


class ParaphrasingGenerator:

    def __init__(self, model_path: str, encoder_name: str):
        self.model_path = model_path
        self.encoder_name = encoder_name
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.encoder = SentenceTransformer(encoder_name)
        self.generator = ParaphrasingPipeline(model=self.model, tokenizer=self.tokenizer)

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def _load_model(self):
        model = GPT2ParaphrasingLM.from_pretrained(self.model_path)
        model.eval()
        return model

    def generate(self, sentence: str, strategy: str="sampling", filter_best: bool=False, k: int=10):
        assert strategy in ("sampling", "beam_search")
        embeddings = self.encoder.encode([sentence], convert_to_tensor=True)
        strategy: Callable = self._sampling if strategy == "sampling" else self._beam_search
        outputs = strategy(1, 256, k, embeddings)
        outputs_sent = {sent.get("generated_text") for sent in outputs if sent != sentence}
        if len(outputs_sent) == 0: return [sentence]
        return self._filter_best(sentence, outputs_sent) if filter_best else outputs_sent

    def _filter_best(self, sentence: str, output_sent: Set[str]):
        original_embedding = normalize(self.encoder.encode([sentence]), axis=1)
        sentences = list(output_sent)
        embeddings = normalize(self.encoder.encode(sentences), axis=1)
        sim = np.matmul(original_embedding, np.transpose(embeddings))
        idx = np.argmax(sim)
        return sentences[idx]

    def _sampling(self, min_len, max_len: int, k: int, embeddings):
        return self.generator("",
            min_length=min_len,
            max_length=max_len,
            do_sample=True,
            repetition_penalty=1.8,
            # add_special_tokens=True,
            num_return_sequences=k,
            temperature=0.8,
            top_p=0.75,
            sentence_embedding=embeddings
        )

    def _beam_search(self, min_len, max_len: int, k: int, embeddings):
        return self.generator("",
            min_length=min_len,
            max_length=max_len,
            repetition_penalty=1.8,
            # add_special_tokens=True,
            num_return_sequences=k,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            sentence_embedding=embeddings
        )


def _process_text(text: str):
    return text.replace("<br />", " ")


if __name__ == '__main__':
    lang = "english"
    encoder_name = "paraphrase-multilingual-mpnet-base-v2"
    model_path = f".cache\checkpoint\gpt2-para-retrined-v3\checkpoint-2520000"
    generator = ParaphrasingGenerator(model_path, encoder_name)

    try:
        while True:
            _input = input('Your sentence: ')
            _input = _process_text(_input)
            pred = generator.generate(_input, strategy="sampling", filter_best=True, k=6)
            print(f'chatbot\'s answer: {pred}')
    except EOFError as e:
        print(end="")

