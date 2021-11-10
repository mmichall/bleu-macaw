import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import LanguageModelingDataset
from model import SentenceVAE

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class LanguageModelingTrainer:

    def __init__(self, dataset: LanguageModelingDataset, model: SentenceVAE, epochs, lr=0.001, batch_size=128, shuffle=True):
        self.dataset = dataset
        self.model = model
        self.NLL = torch.nn.NLLLoss(ignore_index=model.pad_idx, reduction='sum')
        # self.NLL = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_loader = DataLoader(self.dataset, shuffle=shuffle, batch_size=batch_size, pin_memory=True)

    def kl_anneal_function(self, anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == 'linear':
            return min(1, step / x0)

    def loss_fn(self, logp, target, length, mean, logv, anneal_function, step, k, x0):
        # cut-off unnecessary padding from target, and flatten
        # target = target[:, :torch.max(length).item()].contiguous().view(-1)
        # target = target.view(-1)
        # logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood

        NLL_loss = self.NLL(logp, target.to(dtype=torch.long))

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = self.kl_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight

    def fit(self):
        self.model.train()
        step = 0
        for epoch in range(self.epochs):
            for idx, batch in self.data_loader:

                batch = torch.stack(batch).to(device=device, dtype=torch.int)
                # Forward pass
                logp, mean, logv, z = self.model(batch, 80)

                # loss calculation
                NLL_loss, KL_loss, KL_weight = self.loss_fn(logp.reshape(logp.size(0)*logp.size(1), logp.size(2)),
                                                            batch.reshape(batch.size(0)*batch.size(1)),
                                                            self.batch_size, mean, logv,
                                                            'linear', step, 0, 10)

                loss = (NLL_loss + KL_weight * KL_loss) / self.batch_size

                # backward + optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                step += 1

                sys.stdout.write(f'\r+\tNLL_loss: {NLL_loss}, KL_loss: {KL_loss}, Loss: {loss}')

            checkpoint_path = os.path.join('E:', 'checkpoints', "E%i.pytorch" % epoch)
            torch.save(self.model.state_dict(), checkpoint_path)
            print('\n')
            print("Model saved at %s" % checkpoint_path)
