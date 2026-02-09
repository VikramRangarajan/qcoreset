import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from torchvision.datasets import ImageFolder
from transformers import AutoTokenizer, RobertaTokenizer

from .tinyimagenet import TinyImageNet

use_v2 = hasattr(torchvision.transforms, "v2")


class CorruptedImageFolder(ImageFolder):
    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def get_dataset(args, train=True, train_transform=True):
    idx = None
    if args.dataset in ["cifar10", "cifar100"]:
        if args.dataset == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif args.dataset == "cifar100":
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        else:
            raise NotImplementedError
        if args.arch == "vit":
            if train and train_transform:
                transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),  # Resize for ViT
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.Resize(224),  # Resize for ViT
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
        else:
            if train and train_transform:
                transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )

        dataset = torchvision.datasets.__dict__[args.dataset.upper()](
            root=args.data_dir, train=train, transform=transform, download=True
        )

        if train and transform and args.corrupt_ratio > 0:
            dataset, idx = corrupted_data(args, dataset)

    elif args.dataset == "tinyimagenet":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        if train:
            dataset = TinyImageNet(
                root=args.data_dir, split="train", download=True, transform=transform
            )
        else:
            dataset = TinyImageNet(
                root=args.data_dir, split="val", download=True, transform=transform
            )

        if train and transform and args.corrupt_ratio > 0:
            dataset, idx = corrupted_data(args, dataset)

    elif args.dataset == "imagenet":
        use_v2 = hasattr(torchvision.transforms, "v2")
        if use_v2:
            import torchvision.transforms.v2 as T

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        if train:
            train_dir = "your/train"
            if use_v2:
                transform = T.Compose(
                    [
                        T.RandomResizedCrop(224),
                        T.RandomHorizontalFlip(),
                        T.ToImage(),
                        T.ToDtype(torch.float32, scale=True),
                        T.Normalize(mean=imagenet_mean, std=imagenet_std),
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
                    ]
                )
            dataset = CorruptedImageFolder(root=train_dir, transform=transform)

            if args.corrupt_ratio > 0:
                dataset, idx = corrupted_data(args, dataset)

        else:
            val_dir = "your/val"
            if use_v2:
                transform = T.Compose(
                    [
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToImage(),
                        T.ToDtype(torch.float32, scale=True),
                        T.Normalize(mean=imagenet_mean, std=imagenet_std),
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
                    ]
                )
            dataset = CorruptedImageFolder(root=val_dir, transform=transform)

    elif args.dataset == "emnist":
        if train and train_transform:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        dataset = torchvision.datasets.EMNIST(
            root=args.data_dir,
            split="balanced",
            train=train,
            download=True,
            transform=transform,
        )
        if train and transform and args.corrupt_ratio > 0:
            dataset, idx = corrupted_data(args, dataset)

    elif args.dataset == "mnist":
        if train and train_transform:
            transform = transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )

        dataset = torchvision.datasets.MNIST(
            root=args.data_dir, train=train, download=True, transform=transform
        )
        if train and transform and args.corrupt_ratio > 0:
            dataset = corrupted_data(args, dataset)
    elif args.dataset == "svhn":
        if train and train_transform:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        if train:
            dataset = torchvision.datasets.SVHN(
                root=args.data_dir, split="train", download=True, transform=transform
            )
            if train and transform and args.corrupt_ratio > 0:
                dataset = corrupted_data(args, dataset)
        else:
            dataset = torchvision.datasets.SVHN(
                root=args.data_dir, split="test", download=True, transform=transform
            )

    elif args.dataset == "snli":
        snli = load_dataset("snli")
        snli = snli.filter(lambda x: x["label"] != -1)  # Remove ambiguous samples
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        def tokenize_fn(example):
            return tokenizer(
                example["premise"],
                example["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )

        snli = snli.map(tokenize_fn, batched=True)
        snli.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        if train and args.corrupt_ratio > 0:
            dataset, idx = corrupt_labels(snli["train"], args.corrupt_ratio)
        elif train:
            dataset = snli["train"]
        else:
            dataset = snli["validation"]

    elif args.dataset == "trec":
        dataset = load_dataset("trec", trust_remote_code=True)
        label_type = "fine_label"  # or "fine"
        label_list = dataset["train"].features[label_type].names
        num_labels = len(label_list)
        print("Number of classes:", num_labels)

        model_name = "google/electra-small-discriminator"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def preprocess(example):
            return tokenizer(example["text"], truncation=True, padding="max_length")

        dataset = dataset.map(preprocess, batched=True)
        dataset = dataset.rename_column(label_type, "label")
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        if train and args.corrupt_ratio > 0:
            dataset, idx = corrupt_labels(dataset["train"], args.corrupt_ratio)
        elif train:
            dataset = dataset["train"]
        else:
            dataset = dataset["test"]

    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")

    return dataset, idx


def corrupt_labels(dataset, corruption_ratio=0.2, num_labels=3):
    corrupted = dataset.map(lambda x: x)  # copy

    n = len(corrupted)
    corrupt_indices = np.random.choice(
        n, int(corruption_ratio * n), replace=False
    ).astype(int)
    for i in corrupt_indices:
        true_label = corrupted[int(i)]["label"]
        new_label = np.random.choice(
            [lab for lab in range(num_labels) if lab != true_label]
        )
        corrupted[int(i)]["label"] = new_label

    return corrupted, corrupt_indices


def corrupted_data(args, dataset):
    corrupt_ratio = args.corrupt_ratio
    idx = np.random.choice(
        len(dataset), size=int(len(dataset) * corrupt_ratio), replace=False
    ).astype(int)
    if args.dataset == "svhn":
        temp = dataset.labels
    else:
        temp = np.array(dataset.targets)
    change = temp[idx]
    for i, e in enumerate(change):
        test = np.arange(args.num_classes)
        test = np.concatenate([test[0:e], test[e + 1 :]])
        v = np.random.choice(test)
        change[i] = v
    temp[idx] = change
    dataset.targets = temp.tolist()
    return dataset, idx
