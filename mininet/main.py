# Jim Clampffer - 2025
"""
Modularize the training pipeline for small classifier models.
"""

import argparse
import multiprocessing
import torch
import torch.nn as nn
import trainer
from dataloaders import MultiDatasetLoader
from mininet import MiniNet


# Use the whole machine. Safe to assume plenty of memory.
# cpu count assignment ust be done in the parent process before other stuff
# gets set up.
hwthreads = multiprocessing.cpu_count()
torch.set_num_threads(hwthreads)
torch.set_num_interop_threads(hwthreads // 2)


if __name__ == "__main__":
    compute_element = "cuda" if torch.cuda.is_available() else "cpu"

    a = argparse.ArgumentParser("Train a model")
    a.add_argument("--epochs", type=int, default=20)
    a.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar10", "cifar100", "food101"],
    )
    a.add_argument("--batch_size", type=int, default=256)
    a.add_argument("--initial_lr", type=float, default=0.01)
    a.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "silu", "gelu"]
    )
    args = a.parse_args()

    setname = args.dataset.lower()

    dataset = MultiDatasetLoader(
        setname,
        trainer.get_basic_augmentation,
        batch_size=args.batch_size,
        download=True,
    )


    activation_map = {
        'relu':nn.ReLU,
        'silu':nn.SiLU,
        'gelu':nn.GELU,
    }

    model = MiniNet(dataset.num_classes, dataset.img_channels, activation_map[args.activation])

    #model.to(compute_element)

    # make configurable
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainer_job = trainer.ModelTrainer(
        model=model,
        loss_fn=loss_fn,
        dataset=dataset,
        epochs=args.epochs,
        initial_lr=args.initial_lr,
        batch_size=args.batch_size,
        optimizer_cls=torch.optim.AdamW,
    )

    # Start training, trainer handles saving the trained model.
    trainer_job.train()

