# Stable Coresets via Posterior Sampling: Aligning Induced and Full Loss Landscapes
This is the official repository for the paper: **Stable Coresets via Posterior Sampling: Aligning Induced and Full Loss Landscapes**. 

In this work, we propose a novel framework that addresses these limitations. First, we establish a connection between posterior sampling and loss landscapes, enabling robust coreset selection even in high-data-corruption scenarios. Second, we introduce a smoothed loss function based on posterior sampling onto the model weights, enhancing stability and generalization while maintaining computational efficiency. We also present a novel convergence analysis for our sampling-based coreset selection method. Finally, through extensive experiments, we demonstrate how our approach achieves faster training and enhanced generalization across diverse datasets than the current state of the art.

---

## Project Structure

```
stable-coreset/
├── train.py               # Main training script
├── mydatasets/            # Dataset loaders + optional label corruption
├── trainers/              # Different coreset selection methods
├── utils/                 # Logging, metrics, configs
└── README.md
```

---

## Features

- **Coreset selection support** (`--selection_method`)
- **Multiple selection strategies**: random, stable coreset, crest, etc.
- **Dataset support**:
  - CIFAR-10/100, TinyImageNet, ImageNet
  - MNIST, EMNIST, SVHN
  - SNLI, TREC (text classification)
- **Label corruption for robustness evaluation** (`--corrupt_ratio`)
- **Supports ResNet, ViT, and custom models**

---

## Example Usage

### Train normally (no coreset) with 10% label corruption:
```bash
python train.py --selection_method=random_full --dataset=cifar10 --arch=resnet20 --corrupt_ratio=0.1
```

### Train with stable coreset selection under 10% data budget and 10% label corruption:
```bash
python train.py --selection_method=single_spread_bn --dataset=cifar10 --arch=resnet20 --ensemble_num=4 --corrupt_ratio=0.1--noise_std=0.01
```


---

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--selection_method` | Different coreset selection method (`random`, `crest`, `random_full`, `single_spread_bn`)|
| `--train_frac` | Fraction of dataset to keep (e.g., `0.1` = 10%) |
| `--arch` | Model (`resnet18`, `vit`, etc.) |
| `--corrupt_ratio` | Percentage of labels to corrupt randomly |
| `--dataset` | Dataset name (e.g., `cifar10`, `imagenet`) |
| `--data_dir` | The directory to store the dataset. (default: `./data`) (For imagenet, you need to directly modify it in mydataset/datasests)|

---

## Add a New Coreset Method

1. Create a new file in `trainer/`:
```python
class MyTrainer(BaseSelector):
```
2. Register it in the argument parser in `utils/arguments.py`

3. Run:
```bash
python train.py --coreset --selection_method MyTrainer
```

## Add a New dataset

1. Write the data pipeline `mydataset/datasets`.

2. Register it in the argument parser in `utils/arguments.py` 

## Add a New model

1. Create a new file in models.

2. Register it in the argument parser in `utils/arguments.py` 

---

## Authors

| Name | Affiliation | Contact |
|------|-------------|---------|
| Wei-Kai Chang | Purdue University | chang986@purdue.edu |
| Rajiv Khanna  | Purdue University | rajivak@purdue.edu |

---
## License

This project is licensed under the MIT License

## Acknowledgements

- Inspired by prior work on coresets, subset selection, submodular optimization and [Crest](https://github.com/YuYang0901/CREST/tree/main)
- PyTorch-based implementation  
- Uses HuggingFace Datasets for SNLI/TREC support
