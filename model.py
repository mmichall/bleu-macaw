import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class RNNTextParaphrasingModel(nn.Module):

    def __init__(self, sentence_transformer: SentenceTransformer, rnn_size: int, rnn_dropout: float,
                 embedding_dim: int, vocab_size: int, pad_index: int = None, embedding_dropout_rate: float = .0,
                 num_layers: int = 1, bidirectional: bool = False):
        super(RNNTextParaphrasingModel, self).__init__()
        self.rnn_size = rnn_size
        self.embedding_dim = embedding_dim
        self.embedding_dropout_rate = embedding_dropout_rate
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_dropout = rnn_dropout
        self.sentence_transformer = sentence_transformer
        self.pad_index = pad_index
        self.vocab_size = vocab_size

        self.input_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.pad_index,
            device=device
        )
        self.input_embedding_dropout = nn.Dropout(
            p=self.embedding_dropout_rate
        )
        self.decoder_rnn = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.rnn_size,
            num_layers=self.num_layers,
            dropout=rnn_dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
            device=device
        )

        self.embed2hidden = nn.Linear(self.sentence_transformer.get_sentence_embedding_dimension(), self.rnn_size,
                                      device=device)
        # TODO: self.rnn_size * self.num_layers
        self.outputs2vocab = nn.Linear(self.rnn_size, self.vocab_size, device=device)
        self.to(device=device)

    def forward(self, raw, input_sentence, prev_hidden=None):
        if prev_hidden is not None:
            hidden = prev_hidden
        else:
            sentence_embeddings = raw
            # sentence_embeddings = self.sentence_transformer.encode(raw,
            #                                                        convert_to_numpy=False,
            #                                                        convert_to_tensor=True)
            hidden = self.embed2hidden(sentence_embeddings).unsqueeze(0)
            if self.num_layers > 1:
                hidden_layer = hidden
                for _ in range(self.num_layers - 1):
                    hidden = torch.concat((hidden, hidden_layer), dim=0)
        input_embedding = self.input_embedding(input_sentence)
        if self.training or self.embedding_dropout_rate > 0:
            input_embedding = self.input_embedding_dropout(input_embedding)

        outputs, hidden = self.decoder_rnn(input_embedding, hidden)
        logp = nn.functional.log_softmax(self.outputs2vocab(outputs), dim=-1)

        return logp, hidden

    def paraphrase(self, sentence: [str]) -> [int]:
        words = []
        for t in range(30):
            if t == 0:
                #TODO: replace 101 with SOS index
                input_sequence = torch.unsqueeze(torch.Tensor(1).fill_(2).long(), dim=0).to(device=device)
                prev_hidden = None
            x = None
            if sentence:
                x = torch.tensor(self.sentence_transformer.encode(sentence,
                                                                 convert_to_numpy=False,
                                                                 convert_to_tensor=True))
            logp, hidden = self(x, input_sequence, prev_hidden)
            word_index = np.argmax(logp.cpu().detach().numpy())
            words.append(word_index)
            input_sequence = torch.tensor([word_index], device=device)
            input_sequence = torch.unsqueeze(input_sequence, dim=1)
            sentence = None
            prev_hidden = hidden
        return words
