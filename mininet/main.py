# Jim Clampffer - 2025
"""
Modularize a training pipeline for small classifier models.

Components can be used as standalone modules elsewhere.
"""

import argparse
import torch
import torch.nn as nn

import activations
import trainer
from dataloaders import MultiDatasetLoader
from resourceman import ResourceMan
from mininet import MiniNet

# Probe resources before calling torch
platform_info = ResourceMan()

# Use the whole machine. Safe to assume plenty of memory or at least
# easy to patch for a specific platform.
torch.set_num_threads(platform_info.hw_threads)
torch.set_num_interop_threads(platform_info.hw_threads // 2)


def add_cli_basic_args(parser: argparse.ArgumentParser):
    """Just to keep __main__ short enough to fit in a terminal"""
    # I don't usually care about format - but when I do, it's something
    # like this. The pattern makes missing stuff stand out a bit.
    # fmt: off
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="number of training epochs",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="mininet",
        choices=["mininet"],
        help="modify the code in main.py if your model of choice isn't available",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar10", "cifar100", "food101"],
        help="datasets from torchvision 'just work'",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        choices=[64,128,256,512,1024,2048],
        help="number of samples to process per pass",
    )
    
    parser.add_argument(
        "--initial_lr",
        type=float,
        default=0.01,
        help="Initial learning rate, run a sweep to see what works well",   
    )
    
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=[
            "relu", "relu6", "prelu",
            "silu", "swish",
            "gelu",
            "mish",
            "serf",
        ],
        help="activation function - applies to whole model for now",
    )

    parser.add_argument(
        "--squeeze_factor",
        type=int,
        default=8,
        choices=[i for i in range(16)],
    )

    parser.add_argument(
        "--modelpath",
        type=str,
        default="",
        help="Load a model to resume training. Optimizer and scheduler will be reset.",
    )

    parser.add_argument(
        "--loss_label_smoothing",
        type=float,
        default=0.1,
        help="Loss function label smoothing"
    )
    # fmt: on
    return parser


def resolve_activation_name(name: str) -> type[nn.Module]:
    activation_map = {
        "relu": nn.ReLU,
        # relu bounded 0..6
        "relu6": nn.ReLU6,
        # like leaky relu, but without the extra hyperparameter
        "prelu": nn.PReLU,
        # smooth. some negative leak to avoid killing neurons
        "swish": nn.SiLU,
        "silu": nn.SiLU,
        # smooth, backprop makes it to early layers. no discontinuity
        "gelu": nn.GELU,
        # smooth, self regulating, good gradient flow
        "mish": nn.Mish,
        # new and fancy
        "serf": activations.SERF,
    }
    return activation_map[name.lower()]


if __name__ == "__main__":
    cli_arg_parser = argparse.ArgumentParser("Train a model")

    # Basic training args, add more structure later
    cli_arg_parser = add_cli_basic_args(cli_arg_parser)
    # OneCycleLR specific
    cli_arg_parser = trainer.add_scheduler_args(cli_arg_parser)
    args = cli_arg_parser.parse_args()

    setname = args.dataset.lower()

    dataset = MultiDatasetLoader(
        setname,
        trainer.get_basic_augmentation,
        batch_size=args.batch_size,
        resman=platform_info,
        download=True,
    )

    model = None
    if args.modelpath != "":
        model = MiniNet.from_file(args.modelpath, resolve_activation_name)
    else:
        model = MiniNet(
            dataset.num_classes,
            resolve_activation_name(args.activation),
            squeeze_factor=args.squeeze_factor,
        )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.loss_label_smoothing)

    trainer_job = trainer.ModelTrainer(
        model=model,
        loss_fn=loss_fn,
        dataset=dataset,
        epochs=args.epochs,
        cliargs=args,  # fixme: this in addition to params is gross.
        platform_info=platform_info,
        initial_lr=args.initial_lr,
        batch_size=args.batch_size,
    )

    # Start training, trainer handles saving the trained model.
    trainer_job.train()
