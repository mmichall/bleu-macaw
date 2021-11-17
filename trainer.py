import numpy as np
import torch
import sys

from torch import Tensor, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
import os

import config
from dataset import LanguageModelingDataset
from model import RNNTextParaphrasingModel

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class ParaphrasingModelTrainer:

    def __init__(self, dataset: LanguageModelingDataset, model: RNNTextParaphrasingModel, optimizer: optim.Optimizer,
                 criterion: _Loss, epochs=1_000_000, batch_size=32, shuffle=True, word_dropout_rate=0.):
        self.dataset = dataset
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.word_dropout_rate = word_dropout_rate
        self.data_loader = DataLoader(self.dataset, shuffle=shuffle, batch_size=batch_size)

    def fit(self):
        self.model.train()

        for epoch in range(self.epochs):
            rec_losses = []
            step = 0
            steps = len(self.data_loader)
            for idx, raw, input, target in self.data_loader:
                self.optimizer.zero_grad()

                logp, _ = self.model(raw, input)
                loss = self.loss(logp, target)

                loss.backward()
                self.optimizer.step()

                step += 1
                rec_losses.append(loss.item())
                loss = np.round(np.mean(rec_losses), 5)

                if step % 9 == 0:
                    sys.stdout.write(f'\r+\tepoch:{epoch}, batches[{self.batch_size}]:'
                                     f'{step} / {steps} [{np.round(100*step/steps, 3)}%], '
                                     f'{self.criterion.__class__.__name__}: {loss}')

            checkpoint_path = os.path.join(config.checkpoint_path, f'model_{epoch}_{loss}.p')
            torch.save(self.model.state_dict(), checkpoint_path)
            print("Checkpoint saved in %s\n" % checkpoint_path)

    def loss(self, logp, target: Tensor):
        logp = logp.transpose(1, 2)
        _loss = self.criterion(logp, target.to(dtype=torch.long))
        return _loss

