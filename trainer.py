import numpy as np
import torch
import sys

from pytorch_model_summary import summary
from torch import Tensor, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
import os

from transformers.pipelines.base import Dataset

import config
from dataset.dataset import LanguageModelingDataset
from dataset.pile import LanguageModelDataset
from model import RNNTextParaphrasingModel

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class ParaphrasingModelTrainer:

    def __init__(self, dataset: LanguageModelDataset, model: RNNTextParaphrasingModel, optimizer: optim.Optimizer,
                 scheduler, criterion: _Loss, epochs=1_000_000, batch_size=32, shuffle=True, word_dropout_rate=0.):
        self.dataset = dataset
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.word_dropout_rate = word_dropout_rate
        self.data_loader = DataLoader(self.dataset.dataset['train'], shuffle=shuffle, batch_size=batch_size,
                                      drop_last=False)
        self.valid_data_loader = DataLoader(self.dataset.dataset['valid'], batch_size=batch_size)

    def fit(self):
        self.model.train()

        for epoch in range(self.epochs):
            rec_losses = []
            steps = len(self.data_loader)
            for step, batch in enumerate(self.data_loader):
                self.model.train()
                self.optimizer.zero_grad()

                logp, _ = self.model(batch['text'], batch['input_ids'])
                loss = self.loss(logp, batch['target_ids'])

                loss.backward()
                self.optimizer.step()

                rec_losses.append(loss.item())

                if step % 9 == 0:
                    loss = np.round(np.mean(rec_losses), 5)
                    sys.stdout.write(f'\r+\tepoch:{epoch}, batches[{self.batch_size}]:'
                                     f'{step} / {steps} [{np.round(100*step/steps, 3)}%], '
                                     f'{self.criterion.__class__.__name__}: {loss}')

            # self.scheduler.step()
            loss = np.round(np.mean(rec_losses), 5)
            checkpoint_path = os.path.join(config.checkpoint_path, f'model_dropout_{epoch}_{loss}.p')
            torch.save(self.model.state_dict(), checkpoint_path)
            print("\nCheckpoint saved in %s\n" % checkpoint_path)

            # evaluate model:
            self.model.eval()

            with torch.no_grad():
                valid_losses = []
                for _, batch in enumerate(self.valid_data_loader):
                    logp, _ = self.model(batch['text'], batch['input_ids'])
                    loss = self.loss(logp, batch['target_ids'])
                    valid_losses.append(loss.item())
                print(f'\n\tvalidation' f'{self.criterion.__class__.__name__}: {np.round(np.mean(valid_losses), 5)}')

    def loss(self, logp, target: Tensor):
        logp = logp.transpose(1, 2)
        _loss = self.criterion(logp, target)
        return _loss

