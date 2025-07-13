# Jim Clampffer 2025
"""
Modularized training utilities as well as training loop.
"""
import itertools

import torch
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_basic_augmentation(X_DIM=224, Y_DIM=224):
    """
    Returns a basic augmentation pipeline for image datasets.
    This is used in the main script to define the transformation.
    """
    enable_affine = False
    if X_DIM > 100 and Y_DIM > 100:
        # Too much interpolation loss on small images.
        enable_affine = True
    else:
        print("Too small for affine xform")

    affine = []
    if enable_affine:
        affine.append(
            transforms.RandomAffine(
                degrees=8, translate=(0.15, 0.15), scale=(0.85, 1.1), shear=10
            )
        )

    return transforms.Compose(
        [
            transforms.Resize(
                (X_DIM, Y_DIM)
            ),  # Really need to pull this from the dataset.
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
    """Manage the training loop and model state"""

    __slots__ = (
        "device",
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
    )

    def __init__(
        self,
        model,
        loss_fn,
        dataset,
        epochs,
        platform_info,
        initial_lr=0.01,
        batch_size=128,
        optimizer_cls=torch.optim.AdamW,
        checkpoint_path=None,
    ):
        self.platform_info = platform_info
        self.device = "cuda" if platform_info._cuda_enabled == True else "cpu"
        self.model = model.to(self.device)

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
        self.optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=1e-4)
        steps_per_epoch = len(self.dataloader)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-1,
            total_steps=epochs * steps_per_epoch,
            pct_start=0.3,  # LR rise portion
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25,
            final_div_factor=1e4
        )

        self.checkpoint_path = checkpoint_path
        self.use_amp = self.device.startswith("cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        self.start_epoch = 0

    def _save_checkpoint(self, epoch):
        torch.save(self.model, "model-epoch{}.pth".format(epoch))
        # quick check
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("total params: {}, trainable params: {}".format(total_params, trainable_params))

    def train(self):
        per_batch_scheduler = isinstance(
            self.scheduler,
            torch.optim.lr_scheduler._LRScheduler,
        ) or isinstance(self.scheduler, OneCycleLR) or isinstance(self.scheduler, torch.optim.SGD)

        try:
            for epoch in range(self.start_epoch, self.epochs):
                total_loss = 0
                for batch in self.dataloader:
                    inputs, targets = batch
                    if isinstance(inputs, (list, tuple)):
                        inputs = [x.to(self.device, non_blocking=True) for x in inputs]
                    else:
                        inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

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

                self._save_checkpoint(epoch)
                normalized_loss = total_loss / len(self.dataloader.dataset)
                print("epoch {} loss = {}".format(epoch, normalized_loss))
                # avoid cost of full validation every epoch
                self._validate(
                    0.1 if epoch % 10 != 0 else 1.0, "epoch {}".format(epoch)
                )
                print("current learning rate: ".format(self.optimizer.param_groups[0]['lr']))

        except KeyboardInterrupt:
            print("Training interrupted. Saving checkpoint...")
            self._save_checkpoint(epoch)
        except Exception as e:
            print("An error occurred during training: {}".format(e))
            self._save_checkpoint(epoch)
            raise

    def _validate(self, limit_frac: float, tag: str):
        """Run validation on limit_frac * validation set, not randomized."""
        self.model.eval()
        loader = self.validateloader
        if limit_frac < 1.0:
            total_batches = len(loader)
            loader = itertools.islice(loader, int(total_batches * limit_frac))

        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x).argmax(dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        acc = 100 * correct / total
        print("{} validation (≈{} samples): {:.2f}%".format(tag, total, acc))


if __name__ == "__main__":
    pass
