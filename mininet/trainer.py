# Jim Clampffer 2025
"""
Modularized training utilities as well as training loop.
"""

import argparse
import itertools
import time
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as transforms

import torch.nn


import mininet


# todo: tensor dataset wrapper + v2 transforms.
def get_basic_augmentation(X_DIM=224, Y_DIM=224) -> transforms.Compose:
    """
    @brief Build a basic augmentation pipeline for image datasets.

    Affine and random crop transforms are predicated on image size due
    to data loss issues at low resolutions.
    """

    affine = []
    if X_DIM > 100 and Y_DIM > 100:
        affine.append(
            transforms.RandomAffine(
                degrees=8, translate=(0.15, 0.15), scale=(0.85, 1.1), shear=10
            )
        )
    else:
        # Tiny images (e.g. cifar10/cifar100) get trashed after affine
        # and rotation transform. So don't use them.
        print("Skipping affine transform - image too small")

    resize = None
    if X_DIM >= 224 and Y_DIM >= 224:
        # Enable random crop on large-ish images - works well on food101.
        # Need to be big enough to not lose a ton of info due to crop.
        extra_px = 4
        resize = [
            transforms.Resize((X_DIM + extra_px, Y_DIM + extra_px)),
            transforms.RandomCrop((X_DIM, Y_DIM), extra_px),
        ]
    else:
        resize = [transforms.Resize((X_DIM, Y_DIM))]

    return transforms.Compose(
        resize
        + [
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        + affine
        + [
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class ModelTrainer:
    """@brief Manage the training loop and model state"""

    # todo: cleanup, add type annotations.
    __slots__ = (
        "_device",
        "model",
        "loss_fn",
        "epochs",
        "batch_size",
        "optimizer",
        "scheduler",
        "checkpoint_path",
        "use_amp",
        "scaler",
        "dataloader",
        "validateloader",
        "start_epoch",
        "platform_info",
        "cli_args",
    )

    _device: str
    model: mininet.MiniNet  # fixme

    def __init__(
        self,
        model,
        loss_fn,
        dataset,
        epochs,
        cliargs,
        platform_info,
        initial_lr=0.01,
        batch_size=128,
        checkpoint_path=None,
    ):
        self.platform_info = platform_info
        self._device = "cuda" if platform_info._cuda_enabled == True else "cpu"
        self.model = model.to(self._device)
        self.cli_args = cliargs

        executor_count = torch.cuda.device_count()
        if executor_count > 1:
            assert False, "not implemented"
            # scale up, not out for short term
        else:
            self.dataloader = dataset.get_train_loader()
            self.validateloader = dataset.get_val_loader()

        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size

        # This seems to do well generally.
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=1e-4
        )
        steps_per_epoch = len(self.dataloader)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cliargs.oclr_max_lr,
            total_steps=epochs * steps_per_epoch,
            pct_start=cliargs.oclr_pct_start,
            cycle_momentum=True,  # todo: parameterize
            base_momentum=cliargs.oclr_base_momentum,
            max_momentum=cliargs.oclr_max_momentum,
            div_factor=cliargs.oclr_div_factor,
            final_div_factor=cliargs.oclr_final_div_factor,
        )

        self.checkpoint_path = checkpoint_path
        self.use_amp = self._device.startswith("cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        self.start_epoch = 0

    def _save_checkpoint(self, epoch, opt, sched):
        """@brief Save model, weights, and current training state"""
        self.model.to_file(epoch, opt, sched, "model-mininet-e{}.pth".format(epoch))

    def train(self):
        """Run the training loop"""

        # print metadata about model
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            "{} total params: {}, trainable params: {}".format(
                self.model.arch_name, total_params, trainable_params
            )
        )

        # Keep for now. OneCycleLR seems like the best scheduler
        # in testing so far.
        per_batch_scheduler = (
            isinstance(
                self.scheduler,
                torch.optim.lr_scheduler._LRScheduler,
            )
            or isinstance(self.scheduler, OneCycleLR)
            or isinstance(self.scheduler, torch.optim.SGD)
        )
        best_acc: float = 0
        try:
            for epoch in range(self.start_epoch, self.epochs):
                enter_time = time.time()
                total_loss = 0
                for batch in self.dataloader:
                    inputs, targets = batch
                    if isinstance(inputs, (list, tuple)):
                        inputs = [x.to(self._device, non_blocking=True) for x in inputs]
                    else:
                        inputs = inputs.to(self._device, non_blocking=True)
                    targets = targets.to(self._device, non_blocking=True)

                    self.model.train()
                    self.optimizer.zero_grad()
                    with autocast(enabled=self.use_amp):
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs, targets)
                        self.scaler.scale(loss).backward()

                        total_loss += loss.item() * inputs.size(0)

                        if self.use_amp or not torch.cuda.is_available():
                            # Keep gradients sane on CPU and mixed precision GPU
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_norm=1.0
                            )

                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                        if self.scheduler and per_batch_scheduler:
                            self.scheduler.step()
                if self.scheduler and not per_batch_scheduler:
                    self.scheduler.step()

                tx = time.time()
                training_elapsed = tx - enter_time
                enter_time = tx

                # avoid cost of full validation every epoch
                acc = self._validate(
                    0.5 if epoch % 5 != 0 else 1.0, "epoch {}".format(epoch)
                )
                if acc > best_acc:
                    # todo: on shortcut validations run a full validation prior to
                    # updating acc and saving
                    best_acc = acc
                    self._save_checkpoint(epoch, self.optimizer, self.scheduler)

                normalized_loss = total_loss / len(self.dataloader.dataset)
                print("epoch {} loss = {}".format(epoch, normalized_loss))

                print(
                    "current learning rate: {}".format(
                        self.optimizer.param_groups[0]["lr"]
                    )
                )
                validation_elapsed = time.time() - enter_time

                print(
                    "training time: {}, validation time: {}".format(
                        training_elapsed, validation_elapsed
                    )
                )

        except KeyboardInterrupt:
            print("Training interrupted. Saving checkpoint...")
            self._save_checkpoint(epoch, self.optimizer, self.scheduler)
        except Exception as e:
            print("An error occurred during training: {}".format(e))
            self._save_checkpoint(epoch, self.optimizer, self.scheduler)
            raise

    def _validate(self, limit_frac: float, tag: str) -> float:
        """Run validation on limit_frac * validation set, not randomized."""
        self.model.eval()
        loader = self.validateloader
        if limit_frac < 1.0:
            total_batches = len(loader)
            loader = itertools.islice(loader, int(total_batches * limit_frac))

        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self._device), y.to(self._device)
                pred = self.model(x).argmax(dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        acc = 100 * correct / total
        print("{} validation (≈{} samples): {:.2f}%".format(tag, total, acc))
        return acc


def add_scheduler_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--oclr_max_lr",
        type=float,
        default=1e-1,
        help="Sets max_lr for the OneCycleLR scheduler",
    )

    parser.add_argument(
        "--oclr_pct_start", type=float, default=0.3, help="Set OneCycleLR start percent"
    )

    parser.add_argument(
        "--oclr_base_momentum",
        type=float,
        default=0.85,
        help="base momentum for OnceCycleLR - momentum always enabled",
    )

    parser.add_argument(
        "--oclr_max_momentum",
        type=float,
        default=0.95,
        help="set the max oclr scheduler momentum",
    )

    parser.add_argument(
        "--oclr_div_factor",
        type=int,
        default=25,
        help="Starting scheduler rate div factor",
    )

    parser.add_argument(
        "--oclr_final_div_factor",
        type=int,
        default=1e4,
        help="Final fix factor on last epoch",
    )

    return parser


if __name__ == "__main__":
    pass
