import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

import torch.nn.functional as F

import dataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class RNNTextGeneratingModel(nn.Module):

    def __init__(self, sentence_transformer: SentenceTransformer, rnn_size: int, rnn_dropout: float,
                 target_embedding_dim: int, vocab_size: int, pad_index: int = None, target_embedding_dropout: float = .0,
                 num_layers: int = 1, bidirectional: bool = False):
        super(RNNTextGeneratingModel, self).__init__()
        self.rnn_size = rnn_size
        self.target_embedding_dim = target_embedding_dim
        self.target_embedding_dropout = target_embedding_dropout
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_dropout = rnn_dropout
        self.sentence_transformer = sentence_transformer
        self.pad_index = pad_index
        self.vocab_size = vocab_size

        self.target_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.target_embedding_dim,
            padding_idx=self.pad_index,
            device=device
        )
        self.target_embedding_dropout = nn.Dropout(
            p=self.target_embedding_dropout
        )
        self.decoder_rnn = nn.GRU(
            input_size=self.target_embedding_dim,
            hidden_size=self.rnn_size,
            num_layers=self.num_layers,
            dropout=rnn_dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
            device=device
        )

        self.embed2hidden = nn.Linear(self.sentence_transformer.get_sentence_embedding_dimension(), self.target_embedding_dim, device=device)
        self.outputs2vocab = nn.Linear(self.rnn_size * self.num_layers, self.vocab_size, device=device)

        self.to(device=device)

    def forward(self, raw, target_sentence, prev_hidden=None):
        if prev_hidden is not None:
            hidden = prev_hidden
        else:
            sentence_embeddings = self.sentence_transformer.encode(raw, convert_to_numpy=False, convert_to_tensor=True)
            sentence_embeddings = sentence_embeddings.to(device=device)
            hidden = self.embed2hidden(sentence_embeddings)
            hidden = hidden.unsqueeze(0)
        target_embedding = self.target_embedding(target_sentence)
        target_embedding = self.target_embedding_dropout(target_embedding)

        outputs, hidden = self.decoder_rnn(target_embedding, hidden)
        outputs = outputs.contiguous()
        logp = nn.functional.log_softmax(self.outputs2vocab(outputs), dim=-1)

        return logp, hidden

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.rnn_size, device=device),
                torch.zeros(self.num_layers, sequence_length, self.rnn_size, device=device))

    def paraphrase(self, to_paraphrase: [str]):
        words = []
        for t in range(30):
            if t == 0:
                #TODO: replace 2 with SOS index
                input_sequence = torch.unsqueeze(torch.Tensor(1).fill_(2).long(), dim=0).to(device=device)
                prev_hidden = None
            logp, hidden = self(to_paraphrase, input_sequence, prev_hidden)
            word_index = np.argmax(logp.cpu().detach().numpy())
            words.append(word_index)
            input_sequence = torch.tensor([word_index], device=device)
            input_sequence = torch.unsqueeze(input_sequence, dim=1)
            to_paraphrase = None
            prev_hidden = hidden
        return words
