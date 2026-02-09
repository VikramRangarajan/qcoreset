import argparse
import time

import numpy as np
import torch
import trackio as wandb
from torch import nn
from torch.utils.data import DataLoader, Subset

from models import config
from mydatasets import IndexedDataset

from .base_trainer import AverageMeter
from .subset_trainer import SubsetTrainer


class ToggleNoisyLayerNorm(nn.Module):
    def __init__(self, orig_layernorm: nn.LayerNorm, noise_std=0.01):
        super().__init__()
        self.norm = nn.LayerNorm(
            normalized_shape=orig_layernorm.normalized_shape,
            eps=orig_layernorm.eps,
            elementwise_affine=orig_layernorm.elementwise_affine,
        )
        self.noise_std = noise_std
        self.noise_enabled = False

        with torch.no_grad():
            self.norm.weight.copy_(orig_layernorm.weight)
            self.norm.bias.copy_(orig_layernorm.bias)

    def enable_noise(self):
        self.noise_enabled = True

    def disable_noise(self):
        self.noise_enabled = False

    def forward(self, x):
        out = self.norm(x)
        if self.noise_enabled and self.noise_std > 0:
            noise = torch.randn_like(out) * self.noise_std
            out = out + noise
        return out


class single_ensemble(SubsetTrainer):
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
        self.exist_indices = self.train_indices
        self.corrupt = self.train_dataset.corrupt_idx
        self.num_checking = 0
        self.loss_watch = (
            np.ones((self.args.watch_interval, len(self.train_dataset))) * -1
        )
        self.one_hot = np.eye(self.args.num_classes)[self.train_target]
        self.ensemble_num = self.args.ensemble_num

        self.train_output = torch.zeros(
            (self.ensemble_num, len(self.train_dataset), self.args.num_classes),
            device=self.args.device,
        )
        self.train_softmax = torch.zeros_like(self.train_output)
        self.approx_time = AverageMeter()
        self.compare_time = AverageMeter()
        self.similarity_time = AverageMeter()
        if self.args.dataset == "tinyimagenet":
            self.args.random_subset_size = 0.005
            self.train_val_batch_size = int(
                np.ceil(self.args.random_subset_size * self.args.train_size)
            )
        elif self.args.dataset == "snli":
            self.train_val_batch_size = min(
                int(np.ceil(self.args.random_subset_size * self.args.train_size)),
                128 * 3,
            )
        elif self.args.dataset == "imagenet":
            self.args.random_subset_size = 0.002
            self.ensemble_num = 2
        else:
            self.train_val_batch_size = min(
                int(np.ceil(self.args.random_subset_size * self.args.train_size)), 500
            )

        if (
            self.args.dataset == "snli"
            or self.args.dataset == "trec"
            or self.args.arch == "vit"
        ):
            self.replace_layernorm_with_toggleable(noise_std=self.args.noise_std)

    def _train_epoch(self, epoch: int):
        """
        Train the model for one epoch
        :param epoch: current epoch
        """
        if epoch % self.args.select_every == 0:
            select_time = time.time()
            self._select_subset(epoch, len(self.train_loader) * epoch)
            select_total = select_time - time.time()

        self._update_train_loader_and_weights()
        self._reset_metrics()
        lr = self.lr_scheduler.get_last_lr()[0]
        self.args.logger.info(f"Epoch {epoch} LR {lr:.6f}")

        self.model.train()
        config.USE_NOISE = False
        self.train_iter = iter(self.train_loader)
        temp_data_count = 0

        for training_step in range(
            self.steps_per_epoch * epoch, self.steps_per_epoch * (epoch + 1)
        ):
            # check dataset empty
            data_start = time.time()
            try:
                batch = next(self.train_iter)
                temp_data_count += self.args.batch_size
            except StopIteration:
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

            wandb.log(
                {
                    "epoch": epoch,
                    "training_step": training_step,
                    "train_loss": loss.item(),
                    "train_acc": train_acc,
                    "similarity_time": self.similarity_time.avg,
                    "select_forward": self.all_time,
                    "select total time": select_total,
                }
            )

        print("data count:", temp_data_count)

    def _get_train_output(self):
        """
        Evaluate the model on the training set and record the output and softmax
        """
        self.model.eval()

        config.USE_NOISE = True
        if self.args.dataset == "snli" or self.args.dataset == "trec":
            self.toggle_all_noise(enable=True)
        self.forward_time = time.time()
        print(config.USE_NOISE)
        with torch.no_grad():
            for _, (data, _, data_idx) in enumerate(self.train_val_loader):
                if self.args.dataset != "snli" and self.args.dataset != "trec":
                    data = data.to(self.args.device)
                else:
                    data = {k: v.to(self.args.device) for k, v in data.items()}

                if (self.args.arch == "resnet50") or (
                    "fc" not in self.args.selection_method
                ):
                    if self.args.dataset != "snli" and self.args.dataset != "trec":
                        outputs = [self.model(data) for i in range(self.ensemble_num)]
                    else:
                        outputs = [
                            self.model(**data).logits for i in range(self.ensemble_num)
                        ]

                for i in range(self.ensemble_num):
                    self.train_output[i, data_idx] = outputs[i]
                    self.train_softmax[i, data_idx] = outputs[i].softmax(dim=1)

        self.all_time = time.time() - self.forward_time

        if self.args.dataset == "snli" or self.args.dataset == "trec":
            self.toggle_all_noise(enable=False)
        self.model.train()
        config.USE_NOISE = False

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

    def _drop_learned_data(self, epoch: int, training_step: int, indices: np.ndarray):
        """
        Drop the learned data points
        :param epoch: current epoch
        :param training_step: current training step
        :param indices: indices of the data points that have valid predictions
        """

        losses = [
            self.train_criterion(
                torch.from_numpy(self.train_output[i][indices]),
                torch.from_numpy(self.train_target[indices]).long(),
            ).numpy()
            for i in range(self.ensemble_num)
        ]
        mean = sum(losses) / len(losses)
        self.loss_watch[epoch % self.args.watch_interval, indices] = mean

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
        if self.args.dataset != "imagenet":
            subsetsize = int(
                np.ceil(
                    self.args.random_subset_size
                    * self.args.train_size
                    / self.args.num_classes
                )
            )
            print("subset size: ", subsetsize)
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

        else:
            if not hasattr(self, "all_classes"):
                self.all_classes = np.unique(self.train_target)
            if not hasattr(self, "_remaining_classes"):
                self._remaining_classes = list(self.all_classes)
            if not hasattr(self, "_class_to_indices"):
                self._class_to_indices = {
                    c: np.intersect1d(
                        np.where(self.train_target == c)[0],
                        self.train_indices,
                        assume_unique=False,
                    )
                    for c in self.all_classes
                }
            # If not enough classes left to fill batch, reshuffle
            if len(self._remaining_classes) < self.args.batch_size:
                remaining = self._remaining_classes
                extra_needed = self.args.batch_size - len(remaining)
                reshuffled = np.random.permutation(self.all_classes).tolist()
                self._remaining_classes = reshuffled[extra_needed:]
                chosen_classes = remaining + reshuffled[:extra_needed]
            else:
                chosen_classes = self._remaining_classes[: self.args.batch_size]
                self._remaining_classes = self._remaining_classes[
                    self.args.batch_size :
                ]

            subsetsize = 5

            indices = [
                np.random.choice(
                    self._class_to_indices[c], size=subsetsize, replace=False
                )
                for c in chosen_classes
            ]

            self.train_val_batch_size = subsetsize * self.args.batch_size
            return np.concatenate(indices)

    def _select_random_withoutrepeat(self):
        indices = []
        for c in np.unique(self.train_target):
            class_indices = np.intersect1d(
                np.where(self.train_target == c)[0], self.exist_indices
            )
            # redoing the whole selection
            if len(class_indices) < np.ceil(
                self.args.random_subset_size
                * self.args.train_size
                / self.args.num_classes
            ):
                self.exist_indices = self.train_indices
                class_indices = np.intersect1d(
                    np.where(self.train_target == c)[0], self.exist_indices
                )
            indices_per_class = np.random.choice(
                class_indices,
                size=int(
                    np.ceil(
                        self.args.random_subset_size
                        * self.args.train_size
                        / self.args.num_classes
                    )
                ),
                replace=False,
            )
            indices.append(indices_per_class)
        self.exist_indices = np.delete(
            self.exist_indices, np.where(self.exist_indices == indices)
        )
        indices = np.concatenate(indices)
        return indices

    def _select_subset(self, epoch: int, training_step: int):
        """
        Select a subset of the data
        """
        what_happen = time.time()
        super()._select_subset(epoch, training_step)
        self.random_sets = []
        self.subset = []
        self.subset_weights = []
        random_subset_time = time.time()
        for _ in range(self.steps_per_epoch):
            # get a random subset of the data
            if self.args.no_repeat:
                random_subset = self._select_random_withoutrepeat()
            else:
                random_subset = self._select_random_set()
            self.random_sets.append(random_subset)
        random_subset_time = time.time() - random_subset_time
        dataloadertime = time.time()
        self.train_val_loader = DataLoader(
            Subset(self.train_dataset, indices=np.concatenate(self.random_sets)),
            batch_size=self.train_val_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        dataloadertime = dataloadertime - time.time()

        train_output_time = time.time()
        self._get_train_output()
        train_output_time = train_output_time - time.time()
        extra_time = time.time()

        for random_set in self.random_sets:
            # Gather predictions for this subset in bulk (vectorized)
            random_set_tensor = torch.tensor(
                random_set, device=self.train_softmax.device
            )
            preds = self.train_softmax[:, random_set_tensor, :]
            preds = preds.cpu().numpy().copy()
            preds -= self.one_hot[random_set][None, :, :]  # vectorized subtraction
            preds = preds.transpose(1, 0, 2).reshape(len(random_set), -1)

            subset, weight, _, similarity_time = self.subset_generator.generate_subset(
                preds=preds,
                epoch=epoch,
                B=self.args.batch_size,
                idx=random_set,
                targets=self.train_target,
                use_submodlib=(self.args.smtk == 0),
            )

            self.similarity_time.update(similarity_time)

            if self.args.randomparse:
                chosen = np.random.choice(
                    len(subset), size=self.args.batch_size, replace=False
                )
                subset = subset[chosen]
                weight = weight[chosen]

            self.subset.append(subset)
            self.subset_weights.append(weight)

        # Finalize and log
        total_extra_time = time.time() - extra_time
        final_step_time = time.time()
        self.subset = np.concatenate(self.subset)
        self.subset_weights = np.concatenate(self.subset_weights)
        final_step_time = final_step_time - time.time()
        what_happen = time.time() - what_happen
        wandb.log(
            {
                "epoch": epoch,
                "training_step": training_step,
                "subset total extra time": total_extra_time,
                "dataloadertime": dataloadertime,
                "train_output_time": train_output_time,
                "final_step_time": final_step_time,
                "what_happen": what_happen,
                "random_subset_time": random_subset_time,
            }
        )

    def replace_layernorm_with_toggleable(self, noise_std):
        def recursive_replace(module):
            for name, child in module.named_children():
                if isinstance(child, nn.LayerNorm):
                    setattr(
                        module,
                        name,
                        ToggleNoisyLayerNorm(child, noise_std=noise_std).to(
                            self.args.device
                        ),
                    )
                else:
                    recursive_replace(child)

        recursive_replace(self.model)

    def toggle_all_noise(self, enable: bool):
        for module in self.model.modules():
            if isinstance(module, ToggleNoisyLayerNorm):
                if enable:
                    module.enable_noise()
                else:
                    module.disable_noise()
