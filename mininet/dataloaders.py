# Jim Clampffer - 2025
"""
Unified API over datasets from different sources. Only torchvision
builtins for now.

High Priority: DirectoryLoader support for easy way to start building
sets for niche classification tasks.
"""

import resourceman
import torchvision
import torchvision.transforms
import torchvision.datasets
from torch.utils.data import DataLoader
from typing import Any


class MultiDatasetLoader:
    """
    @brief Hide some of the differences in how the training and
           validation subsets are split.

    Augmentation is pushed into the loader since some transforms are
    dependent on resolution.
    """

    __slots__ = (
        # params
        "_dataset_name",
        "_augmentation_transform",
        "_batch_size",
        "_resource_man",
        "_download",
        # driven by param selection
        "_num_classes",
        "_distinct_images",
        "_native_dimensions",
        "_validate_transform",
        # everything else
        "_training_set",
        "_validation_set",
        "_training_loader",
        "_validation_loader",
    )

    # examples: "cifar10", "food101"
    _dataset_name: str
    _batch_size: int
    _resource_man: resourceman.ResourceMan

    # derived
    _num_classes: int
    _distinct_images: int
    _native_dimensions: tuple[int, int, int]
    _validate_transform: torchvision.transforms.Compose
    _augmentation_transform: torchvision.transforms.Compose

    # Cache the loaders once instantiated
    _training_set: Any
    _validation_set: Any
    _training_loader: DataLoader | None
    _validation_loader: DataLoader | None

    def __init__(
        self,
        dataset_name: str,
        transformfactory,
        batch_size: int,
        resman: resourceman.ResourceMan,
        root: str = "./data",
        download: bool = True,
    ):
        self._dataset_name = dataset_name.lower()

        self._batch_size = batch_size
        self._download = download
        self._resource_man = resman
        self._training_loader = None
        self._validation_loader = None

        # derived
        self._num_classes = None

        # Other than cifar use 224x224 for comparison against pretrained models
        if self.dataset_name in ("cifar10", "cifar100"):
            self._native_dimensions = (3, 32, 32)
        else:
            # 224x224 default
            self._native_dimensions = (3, 224, 224)

        _, x, y = self._native_dimensions
        self._augmentation_transform = transformfactory(x, y)

        self._validate_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((x, y)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Need to sort this out. Don't intend to keep adding ifs.
        # See split vs. train in dataset params for rationalle here.
        if self._dataset_name == "cifar10":
            self._num_classes = 10
            self._training_set = torchvision.datasets.CIFAR10(
                root=root,
                train=True,
                transform=self._augmentation_transform,
                download=download,
            )
            self._validation_set = torchvision.datasets.CIFAR10(
                root=root,
                train=False,
                transform=self._validate_transform,
                download=download,
            )
        elif self._dataset_name == "cifar100":
            self._num_classes = 100
            self._training_set = torchvision.datasets.CIFAR100(
                root=root,
                train=True,
                transform=self._augmentation_transform,
                download=download,
            )
            self._validation_set = torchvision.datasets.CIFAR100(
                root=root,
                train=False,
                transform=self._validate_transform,
                download=download,
            )
        elif self._dataset_name == "food101":
            self._num_classes = 101
            self._training_set = torchvision.datasets.Food101(
                root=root,
                split="train",
                transform=self._augmentation_transform,
                download=download,
            )
            self._validation_set = torchvision.datasets.Food101(
                root=root,
                split="test",
                transform=self._validate_transform,
                download=download,
            )
        else:
            raise ValueError("Unsupported dataset name: {}".format(self._dataset_name))

        self._distinct_images = len(self._training_set) + len(self._validation_set)

    def get_transform(self) -> torchvision.transforms.Compose:
        """@brief Get augmentation stack"""
        return self._augmentation_transform

    def _make_loader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            prefetch_factor=64,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self._resource_man.hw_threads // 4 + 1,
        )

    def get_train_loader(self) -> DataLoader:
        """@brief DataLoader for training set"""
        if self._training_loader != None:
            return self._training_loader

        self._training_loader = self._make_loader(self._training_set)
        return self._training_loader

    def get_val_loader(self) -> DataLoader:
        """@brief DataLoader for validation set"""
        if self._validation_loader != None:
            return self._validation_loader

        self._validation_loader = self._make_loader(self._validation_set)
        return self._validation_loader

    def __iter__(self):
        self._train_loader_iter = iter(self.get_train_loader())
        return self

    def __next__(self):
        return next(self._train_loader_iter)

    def __len__(self) -> int:
        """@brief Batches in set for given batch size"""
        return len(self.get_train_loader())

    @property
    def img_channels(self) -> int:
        """@brief Number of channels in the input images."""
        return self._native_dimensions[0]

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def batch_size(self) -> int:
        """Determines batch size of everything downstream"""
        return self._batch_size

    @property
    def dataset_name(self):
        return self._dataset_name
