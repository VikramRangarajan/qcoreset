import logging
import os
import time
from warnings import simplefilter
import copy
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
import timm
from utils import get_args
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from mydatasets import IndexedDataset
from models import *

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

simplefilter(action="ignore", category=FutureWarning)
np.seterr(all="ignore")

args = get_args()
np.random.seed(seed=args.seed)

if len(args.gpu) > 0 and ("CUDA_VISIBLE_DEVICES" not in os.environ):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    device_str = ",".join(map(str, args.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str
    print("Using GPU: {}.".format(os.environ["CUDA_VISIBLE_DEVICES"]))

# Use CUDA if available and set random seed for reproducibility
if torch.cuda.is_available():
    args.device = "cuda"
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    args.device = "cpu"
    torch.manual_seed(args.seed)

if args.use_wandb:
    import wandb

    name = "public version testing"
    wandb.init(project=name, config=args, name=args.save_dir.split("/")[-1])

# Set up logging and output locations
logger = logging.getLogger(
    args.save_dir.split("/")[-1] + time.strftime("-%Y-%m-%d-%H-%M-%S")
)
os.makedirs(args.save_dir, exist_ok=True)

logging.basicConfig(
    filename=f"{args.save_dir}/output.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# define a Handler which writes INFO messages or higher to the sys.stderr
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
args.logger = logger
args.logger.info("Arguments: {}".format(args))
args.logger.info("Time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))


def main(args):
    train_dataset = IndexedDataset(args, train=True, train_transform=True)
    args.train_size = len(train_dataset)
    val_loader = torch.utils.data.DataLoader(
        IndexedDataset(args, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.arch == "resnet20":
        if "single_spread_bn" in args.selection_method:
            model = ResNet20_noise_bn(num_classes=args.num_classes, std=args.noise_std)
        else:
            model = ResNet20(num_classes=args.num_classes)
    elif args.arch == "resnet18":
        if "single_spread_bn" in args.selection_method:
            model = ResNet18_noise_bn(num_classes=args.num_classes, std=args.noise_std)
        else:
            model = ResNet18(num_classes=args.num_classes)
    elif args.arch == "resnet50":
        if args.selection_method == "single_spread_bn":
            model = ResNet50_noise_bn(num_classes=args.num_classes, std=args.noise_std)
        else:
            model = torchvision.models.resnet50(num_classes=args.num_classes)
    elif args.arch == "lenet":
        if args.selection_method == "single_spread":
            model = LeNet(num_classes=args.num_classes, std=args.noise_std)
        else:
            model = LeNet(num_classes=args.num_classes)
    elif args.arch == "roberta":
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=3
        )
        if args.freeze:
            for name, param in model.named_parameters():
                if (
                    "classifier" not in name
                ):  # The 'classifier' is the final classification layer
                    param.requires_grad = False
    elif args.arch == "electra-small-discriminator":
        model = AutoModelForSequenceClassification.from_pretrained(
            "google/electra-small-discriminator", num_labels=args.num_classes
        )
        if args.freeze:
            for name, param in model.named_parameters():
                if (
                    "classifier" not in name
                ):  # The 'classifier' is the final classification layer
                    param.requires_grad = False
    elif args.arch == "vit":  # only for cifar10 and cifar100
        if args.pretrain_vit:
            model = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=args.num_classes
            )
        else:
            model = timm.create_model(
                "vit_base_patch16_224", pretrained=False, num_classes=args.num_classes
            )
    else:
        raise NotImplementedError(f"Architecture {args.arch} not implemented.")

    if args.selection_method == "none":
        from trainers import BaseTrainer

        trainer = BaseTrainer(
            args,
            model,
            train_dataset,
            val_loader,
        )
    elif args.selection_method == "random" or args.selection_method == "random_full":
        from trainers import RandomTrainer

        trainer = RandomTrainer(
            args,
            model,
            train_dataset,
            val_loader,
        )
    elif args.selection_method == "crest":
        from trainers import CRESTTrainer

        trainer = CRESTTrainer(
            args,
            model,
            train_dataset,
            val_loader,
        )
    elif "single_spread" in args.selection_method:
        from trainers import single_ensemble

        trainer = single_ensemble(
            args,
            model,
            train_dataset,
            val_loader,
        )

    else:
        raise NotImplementedError(
            f"Selection method {args.selection_method} not implemented."
        )

    trainer.train()


if __name__ == "__main__":
    main(args)
