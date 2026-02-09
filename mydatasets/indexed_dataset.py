from torch.utils.data import Dataset
from mydatasets.datasets import get_dataset


def cache_dataset(dataset):
    cache = {}
    for i in range(len(dataset)):
        cache[i] = dataset[i]
    return cache


class IndexedDataset(Dataset):
    def __init__(self, args, train=True, train_transform=False):
        super().__init__()
        self.dataset, self.corrupt_idx = get_dataset(
            args, train=train, train_transform=train_transform
        )
        self.args = args

    def __getitem__(self, index):
        if self.args.dataset == "snli" or self.args.dataset == "trec":
            example = self.dataset[int(index)]
            data = {
                "input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"],
            }
            target = example["label"]
        else:
            data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

    def clean(self):
        self._cachers = []
