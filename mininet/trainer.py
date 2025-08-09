# Jim Clampffer 2025
"""
Modularized training utilities as well as training loop.
"""

import argparse
import itertools
import json
import time
import os
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from typing import Optional
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
        # DDP state
        "_world_size",
        "_rank",
        "_local_rank",
        "_ddp_enabled",
        "_train_sampler",
        "_val_sampler",
    )

    _device: str
    model: mininet.MiniNet  # fixme
    
    # DDP state
    _world_size: int
    _rank: int
    _local_rank: int
    _ddp_enabled: bool
    #_train_sampler: Optional[DistributedSampler] = None
    #_val_sampler: Optional[DistributedSampler] = None


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
        #self._device = "cuda" if platform_info._cuda_enabled == True else "cpu"

        self._ddp_enabled = False
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self._rank = int(os.environ.get("RANK", "0"))
        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        using_cuda = platform_info._cuda_enabled

        #self.model = model.to(self._device)

        if using_cuda and self._world_size > 1:
            # Ensure process group is initialized once.
            if not (dist.is_available() and dist.is_initialized()):
                dist.init_process_group(backend="nccl", timeout=torch.distributed.timedelta(seconds=300))
            self._ddp_enabled = True
            self._device = f"cuda:{self._local_rank}"
            torch.cuda.set_device(self._local_rank)
            self.model = model.to(self._device)
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[self._local_rank], output_device=self._local_rank, find_unused_parameters=False)
        else:
            # CPU or single-GPU path (no torchrun)
            if using_cuda and torch.cuda.device_count() > 1 and self._world_size == 1:
                # Continue single-GPU and log a hint
                if self._rank == 0:
                    print("Multiple GPUs detected but running single-GPU mode. To use all GPUs, launch with:")
                    print("  torchrun --standalone --nproc_per_node=<NUM_GPUS> main.py <args>")
            self._device = "cuda" if using_cuda else "cpu"
            self.model = model.to(self._device)

        self.cli_args = cliargs

        # Build loaders (DDP uses DistributedSampler; otherwise unchanged)
        self._train_sampler =  None
        self._val_sampler = None

        if self._ddp_enabled:
            # Per-rank train sampler (shuffle True via set_epoch)
            self._train_sampler = DistributedSampler(dataset._training_set, num_replicas=self._world_size, rank=self._rank, shuffle=True, drop_last=False)
            self.dataloader = dataset.get_train_loader(sampler=self._train_sampler)

            # Validation: default rank0-only (clean, non-distributed)
            if getattr(self.cli_args, "distributed_validate", False):
                # Distributed validation with deterministic order (shuffle=False)
                self._val_sampler = DistributedSampler(dataset._validation_set, num_replicas=self._world_size, rank=self._rank, shuffle=False, drop_last=False)
                self.validateloader = dataset.get_val_loader(sampler=self._val_sampler)
            else:
                if self._rank == 0:
                    self.validateloader = dataset.get_val_loader(sampler=None)
                else:
                    self.validateloader = None
        else:
            # Original single-process loaders
            self.dataloader = dataset.get_train_loader(sampler=None)
            self.validateloader = dataset.get_val_loader(sampler=None)

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
        self.use_amp = str(self._device).startswith("cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        self.start_epoch = 0


        # Compute banner (rank0 only)
        if (not self._ddp_enabled) or (self._ddp_enabled and self._rank == 0):
            compute = {}
            if not using_cuda:
                compute = {"cpu": self.platform_info.hw_threads, "gpu": None}
            elif self._ddp_enabled:
                compute = {
                    "gpu": self._world_size,
                    "model": self.platform_info.gpu_model,
                    "world_size": self._world_size,
                    "rank": self._rank,
                    "local_rank": self._local_rank,
                }
            else:
                compute = {"gpu": 1, "model": self.platform_info.gpu_model, "world_size": 1, "rank": 0, "local_rank": 0}
            print(f"compute={compute}")

    def _save_checkpoint(self, epoch, opt, sched):
        """@brief Save model, weights, and current training state"""
        #self.model.to_file(epoch, opt, sched, "model-mininet-e{}.pth".format(epoch))

        # DDP: save only on rank0; unwrap DDP module if needed
        if self._ddp_enabled and self._rank != 0:
            return
        model_to_save = self.model
        if hasattr(self.model, "module"):
            model_to_save = self.model.module
        model_to_save.to_file(epoch, opt, sched, "model-mininet-e{}.pth".format(epoch))
 


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


                # Ensure epoch-based shuffling in DDP
                if self._train_sampler is not None:
                    self._train_sampler.set_epoch(epoch)

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

                ## avoid cost of full validation every epoch
                #acc = self._validate(
                #    1.0, "epoch {}".format(epoch)
                #)

                # Validation
                if self.validateloader is not None:
                    # Default: rank0-only OR distributed per flag
                    if self._ddp_enabled and getattr(self.cli_args, "distributed_validate", False):
                        acc = self._validate_distributed("epoch {}".format(epoch))
                    else:
                        acc = self._validate(1.0, "epoch {}".format(epoch))
                else:
                    acc = None





                if acc is not None and acc > best_acc:
                    # todo: on shortcut validations run a full validation prior to
                    # updating acc and saving
                    best_acc = acc
                    self._save_checkpoint(epoch, self.optimizer, self.scheduler)

                # Only rank0 writes epoch json/logs
                if (not self._ddp_enabled) or (self._ddp_enabled and self._rank == 0):
                    epoch_data = {
                        "model_arch": self.model.module.arch_name if hasattr(self.model, "module") else self.model.arch_name,
                        "epoch": epoch,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "acc": acc,
                    }
                    json.dump(epoch_data, open("epoch-{}.json".format(epoch), "w"))
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

    def _validate_distributed(self, tag: str) -> float:
        """Run distributed validation with metric all-reduce; clean transforms already applied."""
        assert self.validateloader is not None, "Distributed validation requires a loader"
        self.model.eval()
        correct_local = torch.tensor(0, device=self._device, dtype=torch.long)
        total_local = torch.tensor(0, device=self._device, dtype=torch.long)
        with torch.no_grad():
            for x, y in self.validateloader:
                x, y = x.to(self._device, non_blocking=True), y.to(self._device, non_blocking=True)
                pred = self.model(x).argmax(dim=1)
                total_local += y.size(0)
                correct_local += (pred == y).sum()
        # All-reduce to get global sums
        dist.all_reduce(correct_local, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_local, op=dist.ReduceOp.SUM)
        # Compute accuracy on rank0; other ranks return None
        if self._rank == 0:
            acc = 100.0 * correct_local.item() / max(1, total_local.item())
            print("{} validation (global {} samples): {:.2f}%".format(tag, total_local.item(), acc))
            return acc
        return None


    def _validate(self, limit_frac: float, tag: str) -> float:
        """Run validation on limit_frac * validation set, not randomized."""
        self.model.eval()
        loader = self.validateloader
        if loader is None:
            return None


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
        help="Final div factor on last epoch",
    )

    # DDP related toggles
    parser.add_argument(
        "--distributed_validate",
        action="store_true",
        help="Enable distributed validation with global metric aggregation (default rank0-only).",
    )
    parser.add_argument(
        "--dist_debug",
        action="store_true",
        help="Enable additional per-rank debug logs (e.g., device, ranks).",
    )
    return parser


if __name__ == "__main__":
    pass
