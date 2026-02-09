# a trainer class that inherits from BaseTrainer and only trains on a subset of the data selected at the beginning of every epoch
import argparse
import sys
import time
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import trackio as wandb
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mydatasets import IndexedDataset, SubsetGenerator

from .base_trainer import BaseTrainer


class SubsetTrainer(BaseTrainer):
    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        train_dataset: IndexedDataset,
        val_loader: DataLoader,
        train_weights: torch.Tensor | None = None,
    ):
        super().__init__(args, model, train_dataset, val_loader, train_weights)
        if args.dataset == "svhn":
            self.train_target = self.train_dataset.dataset.labels
        elif args.dataset == "snli" or args.dataset == "trec":
            self.train_target = (
                self.train_dataset.dataset["label"].cpu().detach().numpy()
            )
        else:
            self.train_target = np.array(self.train_dataset.dataset.targets)
        self.subset_generator = SubsetGenerator(
            greedy=(args.selection_method != "rand"), smtk=args.smtk
        )
        self.datasize = len(train_dataset)
        self.num_selection = 0

    def _select_from_subset(self) -> np.ndarray:
        indices = []
        for c in np.unique(self.train_target):
            class_indices = np.intersect1d(
                np.where(self.train_target == c)[0], self.unique
            )
            indices_per_class = np.random.choice(
                class_indices,
                size=int(
                    np.ceil(
                        self.datasize * self.args.train_frac / self.args.num_classes
                    )
                ),
                replace=False,
            )
            indices.append(indices_per_class)
        indices = np.concatenate(indices)

        return indices

    def _update_train_loader_and_weights(self):
        self.train_weights = np.zeros(len(self.train_dataset))
        self.subset_weights = (
            self.subset_weights / np.sum(self.subset_weights) * len(self.subset)
        )
        self.train_weights[self.subset] = self.subset_weights
        self.train_weights = (
            torch.from_numpy(self.train_weights).float().to(self.args.device)
        )
        self.args.logger.info(
            "Updating train loader and weights with subset of size {}".format(
                len(self.subset)
            )
        )

        self.train_loader = DataLoader(
            Subset(self.train_dataset, self.subset),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.train_val_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def _train_epoch(self, epoch):
        # select a subset of the data
        self._select_subset(epoch, len(self.train_loader) * epoch)
        self._update_train_loader_and_weights()

        self.model.train()
        self._reset_metrics()

        data_start = time.time()
        # use tqdm to display a smart progress bar
        pbar = tqdm(
            enumerate(self.train_loader), total=len(self.train_loader), file=sys.stdout
        )
        for batch_idx, (data, target, data_idx) in pbar:
            # load data to device and record data loading time
            if self.args.dataset != "snli" and self.args.dataset != "trec":
                data, target = data.to(self.args.device), target.to(self.args.device)
            else:
                data = {k: v.to(self.args.device) for k, v in data.items()}
                target = target.to(self.args.device)
            data_time = time.time() - data_start
            self.batch_data_time.update(data_time)

            self.optimizer.zero_grad()

            # train model with the current batch and record forward and backward time
            loss, train_acc = self._forward_and_backward(data, target, data_idx)

            data_start = time.time()

            # update progress bar
            pbar.set_description(
                "Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}".format(
                    epoch,
                    self.args.epochs,
                    batch_idx * self.args.batch_size + len(data),
                    len(self.train_loader.dataset),  # type: ignore
                    100.0 * (batch_idx + 1) / len(self.train_loader),
                    loss.item(),
                    train_acc,
                )
            )

        if self.args.cache_dataset and self.args.clean_cache_iteration:
            self.train_dataset.clean()
            self._update_train_loader_and_weights()

    def _get_train_output(self):
        """
        Evaluate the model on the training set and record the output and softmax
        """
        self.model.eval()

        self.train_output = np.zeros((len(self.train_dataset), self.args.num_classes))
        self.train_softmax = np.zeros((len(self.train_dataset), self.args.num_classes))

        with torch.no_grad():
            for _, (data, _, data_idx) in enumerate(self.train_val_loader):
                if self.args.dataset != "snli" and self.args.dataset != "trec":
                    data = data.to(self.args.device)
                    output = self.model(data)
                else:
                    data = {k: v.to(self.args.device) for k, v in data.items()}
                    output = self.model(**data).logits

                self.train_output[data_idx] = output.cpu().numpy()
                self.train_softmax[data_idx] = output.softmax(dim=1).cpu().numpy()

        self.model.train()

    def _select_subset(self, epoch: int, training_step: int):
        """
        Select a subset of the data
        """
        self.num_selection += 1

        if self.args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "training_step": training_step,
                    "num_selection": self.num_selection,
                }
            )
        self.subset: Sequence[int]
        pass

    def _get_num_selection(self):
        return self.num_selection
