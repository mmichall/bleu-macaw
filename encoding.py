import torch
from torch import Tensor


def batch2batch_ohe(batch: Tensor, categories: int = 10_000):
    zeros = torch.zeros((*batch.size(), categories))
    zeros = zeros