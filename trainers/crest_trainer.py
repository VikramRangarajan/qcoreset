import argparse
import time

import numpy as np
import torch
import trackio as wandb
from torch import nn
from torch.utils.data import DataLoader, Subset

from mydatasets import IndexedDataset
from utils import Adahessian

from .base_trainer import AverageMeter
from .subset_trainer import SubsetTrainer


class CRESTTrainer(SubsetTrainer):
    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        train_dataset: IndexedDataset,
        val_loader: DataLoader,
        train_weights: torch.Tensor = None,
    ):
        super().__init__(args, model, train_dataset, val_loader, train_weights)
        self.train_indices = np.arange(len(self.train_dataset))
        self.steps_per_epoch = np.ceil(
            int(len(self.train_dataset) * self.args.train_frac) / self.args.batch_size
        ).astype(int)
        self.reset_step = self.steps_per_epoch
        self.random_sets = np.array([])

        self.num_checking = 0
        self.delta = None
        self.gradient_approx_optimizer = Adahessian(self.model.parameters())

        self.loss_watch = (
            np.ones((self.args.watch_interval, len(self.train_dataset))) * -1
        )

        self.approx_time = AverageMeter()
        self.compare_time = AverageMeter()
        self.similarity_time = AverageMeter()
        # chnage efficient calculation for kernel
        if self.args.dataset == "snli" or self.args.dataset == "trec":
            self.model.config._attn_implementation = "eager"

    def _train_epoch(self, epoch: int):
        """
        Train the model for one epoch
        :param epoch: current epoch
        """
        self.model.train()
        self._reset_metrics()

        lr = self.lr_scheduler.get_last_lr()[0]
        self.args.logger.info(f"Epoch {epoch} LR {lr:.6f}")

        for training_step in range(
            self.steps_per_epoch * epoch, self.steps_per_epoch * (epoch + 1)
        ):
            start_epoch = time.time()
            # checking at each update
            if (training_step > self.reset_step) and (
                (training_step - self.reset_step) % self.args.check_interval == 0
            ):
                self._check_approx_error(epoch, training_step)

            if training_step == self.reset_step:
                self._select_subset(epoch, training_step)
                self._update_train_loader_and_weights()
                self.train_iter = iter(self.train_loader)
                self._get_quadratic_approximation(epoch, training_step)
            elif training_step == 0:
                self.train_iter = iter(self.train_loader)

            # check dataset empty
            data_start = time.time()
            try:
                batch = next(self.train_iter)
            except StopIteration:
                if self.args.cache_dataset and self.args.clean_cache_iteration:
                    self.train_dataset.clean()
                    self._update_train_loader_and_weights()
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)

            data, target, data_idx = batch
            if self.args.dataset != "snli" and self.args.dataset != "trec":
                data, target = data.to(self.args.device), target.to(self.args.device)
            else:
                data = {k: v.to(self.args.device) for k, v in data.items()}
                target = target.to(self.args.device)
            data_time = time.time() - data_start
            self.batch_data_time.update(data_time)

            loss, train_acc = self._forward_and_backward(data, target, data_idx)

            data_start = time.time()
            start_epoch = time.time() - start_epoch
            if self.args.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "training_step": training_step,
                        "train_loss": loss.item(),
                        "train_acc": train_acc,
                        "foward_time": self.batch_forward_time.avg,
                        "time pass from last step": start_epoch,
                    }
                )

    def _forward_and_backward(self, data, target, data_idx):
        self.optimizer.zero_grad()

        # train model with the current batch and record forward and backward time
        forward_start = time.time()
        if self.args.dataset != "snli" and self.args.dataset != "trec":
            output = self.model(data)
        else:
            output = self.model(**data).logits
        forward_time = time.time() - forward_start
        self.batch_forward_time.update(forward_time)

        loss = self.train_criterion(output, target)
        loss = (loss * self.train_weights[data_idx]).mean()

        lr = self.lr_scheduler.get_last_lr()[0]
        if lr > 0:
            # compute the parameter change delta
            self.model.zero_grad()
            # approximate with hessian diagonal
            loss.backward(create_graph=True)
            gf_current, _, _ = self.gradient_approx_optimizer.step(momentum=False)
            if self.delta is None:
                self.delta = 0
            self.delta -= lr * gf_current

        backward_start = time.time()
        loss.backward()
        self.optimizer.step()
        backward_time = time.time() - backward_start
        self.batch_backward_time.update(backward_time)

        # update training loss and accuracy
        train_acc = (output.argmax(dim=1) == target).float().mean().item()
        if self.args.dataset != "snli" and self.args.dataset != "trec":
            self.train_loss.update(loss.item(), data.size(0))
            self.train_acc.update(train_acc, data.size(0))
        else:
            self.train_loss.update(loss.item(), output.shape[0])
            self.train_acc.update(train_acc, output.shape[0])

        return loss, train_acc

    def _get_quadratic_approximation(self, epoch: int, training_step: int):
        """
        Compute the quadratic approximation of the loss function
        :param epoch: current epoch
        :param training_step: current training step
        """

        if self.args.approx_with_coreset:
            # Update the second-order approximation with the coreset
            approx_loader = DataLoader(
                Subset(self.train_dataset, indices=self.subset),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                sampler=None,
            )
        else:
            # Update the second-order approximation with random subsets
            approx_loader = DataLoader(
                Subset(self.train_dataset, indices=self.random_sets),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                sampler=None,
            )

        approx_start = time.time()
        curvature_norm = AverageMeter()
        self.start_loss = AverageMeter()
        for approx_batch, (input, target, idx) in enumerate(approx_loader):
            target = target.to(self.args.device)
            if self.args.dataset == "snli" or self.args.dataset == "trec":
                input_var = {k: v.to(self.args.device) for k, v in input.items()}
                output = self.model(**input_var).logits
            else:
                input_var = input.to(self.args.device)
                output = self.model(input_var)
            target_var = target

            if self.args.approx_with_coreset:
                loss = self.train_criterion(output, target_var)
                batch_weight = self.train_weights[idx.long()]
                loss = (loss * batch_weight).mean()
            else:
                loss = self.val_criterion(output, target_var)
            self.model.zero_grad()

            # approximate with hessian diagonal
            loss.backward(create_graph=True)
            gf_tmp, ggf_tmp, ggf_tmp_moment = self.gradient_approx_optimizer.step(
                momentum=True
            )

            if approx_batch == 0:
                self.gf = gf_tmp * len(idx)
                self.ggf = ggf_tmp * len(idx)
                self.ggf_moment = ggf_tmp_moment * len(idx)
            else:
                self.gf += gf_tmp * len(idx)
                self.ggf += ggf_tmp * len(idx)
                self.ggf_moment += ggf_tmp_moment * len(idx)

            curvature_norm.update(ggf_tmp_moment.norm())
            self.start_loss.update(loss.item(), output.size(0))

        approx_time = time.time() - approx_start
        self.approx_time.update(approx_time)

        self.gf /= len(approx_loader.dataset)
        self.ggf /= len(approx_loader.dataset)
        self.ggf_moment /= len(approx_loader.dataset)
        self.delta = 0

        gff_norm = curvature_norm.avg
        self.start_loss = self.start_loss.avg
        if self.args.approx_moment:
            self.ggf = self.ggf_moment

        if training_step == self.steps_per_epoch:
            self.init_curvature_norm = gff_norm
        else:
            self.args.check_interval = int(
                torch.ceil(self.init_curvature_norm / gff_norm * self.args.interval_mul)
            )
            self.args.num_minibatch_coreset = min(
                self.args.check_interval * self.args.batch_num_mul, self.steps_per_epoch
            )
        self.args.logger.info(
            f"Checking interval {self.args.check_interval}. Number of minibatch coresets {self.args.num_minibatch_coreset}"
        )
        if self.args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "training_step": training_step,
                    "ggf_norm": gff_norm,
                    "check_interval": self.args.check_interval,
                    "num_minibatch_coreset": self.args.num_minibatch_coreset,
                    "approx_time": self.approx_time.avg,
                }
            )

    def _check_approx_error(self, epoch: int, training_step: int) -> torch.Tensor:
        """
        Check the approximation error of the current batch
        :param epoch: current epoch
        :param training_step: current training step
        """
        self.num_checking += 1
        start_compare = time.time()
        self._get_train_output()
        train_output_time = time.time() - start_compare
        true_loss = self.val_criterion(
            torch.from_numpy(self.train_output[self.random_sets]),
            torch.from_numpy(self.train_target[self.random_sets]).long(),
        )

        delta_norm = torch.norm(self.delta)

        approx_loss = torch.matmul(self.delta, self.gf) + self.start_loss
        approx_loss += 1 / 2 * torch.matmul(self.delta * self.ggf, self.delta)

        loss_diff = abs(true_loss - approx_loss.item())
        thresh = self.args.check_thresh_factor * true_loss

        log_str = f"Iter {training_step} loss difference {loss_diff:.3f} threshold {thresh:.3f} True loss {true_loss:.3f} Approx loss {approx_loss.item():.3f} Delta norm {delta_norm:.3f}"

        if loss_diff > thresh:
            self.reset_step = training_step
            log_str += f" is larger than threshold {thresh:.3f}. "
        self.args.logger.info(log_str)

        compare_time = time.time() - start_compare
        self.compare_time.update(compare_time)

        if self.args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "training_step": training_step,
                    "loss_diff": loss_diff,
                    "loss_thresh": thresh,
                    "delta_norm": delta_norm,
                    "num_checking": self.num_checking,
                    "compare_time": compare_time,
                    "train_output_time": train_output_time,
                }
            )

    def _drop_learned_data(self, epoch: int, training_step: int, indices: np.ndarray):
        """
        Drop the learned data points
        :param epoch: current epoch
        :param training_step: current training step
        :param indices: indices of the data points that have valid predictions
        """

        self.loss_watch[epoch % self.args.watch_interval, indices] = (
            self.train_criterion(
                torch.from_numpy(self.train_output[indices]),
                torch.from_numpy(self.train_target[indices]).long(),
            ).numpy()
        )

        if (epoch + 1) % self.args.drop_interval == 0:
            order_ = np.where(
                np.sum(self.loss_watch > self.args.drop_thresh, axis=0) > 0
            )[0]
            unselected = np.where(np.sum(self.loss_watch >= 0, axis=0) == 0)[0]
            order_ = np.concatenate([order_, unselected])

            order = []
            per_class_size = int(
                np.ceil(
                    self.args.random_subset_size
                    * self.args.train_size
                    / self.args.num_classes
                )
            )
            for c in np.unique(self.train_target):
                class_indices_new = np.intersect1d(
                    np.where(self.train_target == c)[0], order_
                )
                if len(class_indices_new) > per_class_size:
                    order.append(class_indices_new)
                else:
                    class_indices = np.intersect1d(
                        np.where(self.train_target == c)[0], self.train_indices
                    )
                    order.append(class_indices)
            order = np.concatenate(order)

            if len(order) > self.args.min_train_size:
                self.train_indices = order

            if self.args.use_wandb:
                wandb.log(
                    {"epoch": epoch, "forgettable_train": len(self.train_indices)}
                )

    def _select_random_set(self) -> np.ndarray:
        subsetsize = int(
            np.ceil(
                self.args.random_subset_size
                * self.args.train_size
                / self.args.num_classes
            )
        )
        if self.args.dataset == "snli":
            subsetsize = min(128, subsetsize)

        # Precompute class -> available indices if not done already
        if not hasattr(self, "_class_to_indices"):
            self._class_to_indices = {
                c: np.intersect1d(
                    np.where(self.train_target == c)[0],
                    self.train_indices,
                    assume_unique=False,
                )
                for c in np.unique(self.train_target)
            }

        indices = [
            np.random.choice(
                self._class_to_indices[c],
                size=min(subsetsize, len(self._class_to_indices[c])),
                replace=False,
            )
            for c in self._class_to_indices
        ]
        return np.concatenate(indices)

    def _select_subset(self, epoch: int, training_step: int):
        """
        Select a subset of the data
        """
        super()._select_subset(epoch, training_step)
        torch.cuda.empty_cache()
        # get random subsets
        self.random_sets = []
        self.subset = []
        self.subset_weights = []
        start = time.time()
        for _ in range(self.args.num_minibatch_coreset):
            # get a random subset of the data
            random_subset = self._select_random_set()
            self.random_sets.append(random_subset)

        self.train_val_loader = DataLoader(
            Subset(self.train_dataset, indices=np.concatenate(self.random_sets)),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self._get_train_output()

        # drop the learned data points
        if self.args.drop_learned:
            self._drop_learned_data(
                epoch, training_step, np.concatenate(self.random_sets)
            )

        for random_set in self.random_sets:
            preds = self.train_softmax[random_set]
            print(f"Epoch [{epoch}] [Greedy], pred size: {np.shape(preds)}")
            if np.shape(preds)[-1] == self.args.num_classes:
                preds -= np.eye(self.args.num_classes)[self.train_target[random_set]]

            (
                subset,
                weight,
                _,
                similarity_time,
            ) = self.subset_generator.generate_subset(
                preds=preds,
                epoch=epoch,
                B=self.args.batch_size,
                idx=random_set,
                targets=self.train_target,
                use_submodlib=(self.args.smtk == 0),
            )
            self.similarity_time.update(similarity_time)

            self.subset.append(subset)
            self.subset_weights.append(weight)

        self.subset = np.concatenate(self.subset)
        self.subset_weights = np.concatenate(self.subset_weights)
        self.random_sets = np.concatenate(self.random_sets)
        start = time.time() - start
        wandb.log({"epoch": epoch, "step": training_step, "select time": start})
