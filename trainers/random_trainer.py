import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mydatasets import IndexedDataset

from .subset_trainer import SubsetTrainer


class RandomTrainer(SubsetTrainer):
    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        train_dataset: IndexedDataset,
        val_loader: DataLoader,
        train_weights: torch.Tensor = None,
    ):
        super().__init__(args, model, train_dataset, val_loader, train_weights)
        self.args = args

    def _select_subset(self, epoch, training_steps):
        # select a subset of the data
        self.num_selection += 1
        if self.args.selection_method == "random_full":
            self.subset = np.random.choice(
                len(self.train_dataset),
                size=int(len(self.train_dataset)),
                replace=False,
            )
        else:
            self.subset = np.random.choice(
                len(self.train_dataset),
                size=int(len(self.train_dataset) * self.args.train_frac),
                replace=False,
            )
        self.subset_weights = np.ones(len(self.subset))
