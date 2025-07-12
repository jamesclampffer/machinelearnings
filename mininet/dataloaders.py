# Jim Clampffer - 2025
"""
Unified API over datasets from different sources. Only torchvision
builtins for now.
"""

import torchvision
import torchvision.datasets.food101 as food101
from torch.utils.data import DataLoader


class MultiDatasetLoader:
    """
    Hide some of the differences in getting the training subset.
    Augmentation is pushed into the loader since some transforms are dependent on resolution.
    """

    __slots__ = (
        "dataset_name",
        "transform",
        "_batch_size",
        "download",
        "native_dimensions",
        "train_set",
        "val_set",
        "distinct_images",
        "_num_classes",
        "_resource_man",
    )
    dataset_name: str
    _batch_size: int
    download: bool
    _num_classes: int

    def __init__(
        self,
        dataset_name,
        transformfactory,
        batch_size,
        resman,
        root="./data",
        download=True,
    ):
        self.dataset_name = dataset_name.lower()
        self._batch_size = batch_size
        self.download = download
        self._resource_man = resman

        # Derived from dataset and common resolutions, need to better wrap this
        self._num_classes = None

        if self.dataset_name in ("cifar10", "cifar100"):
            self.transform = transformfactory(32, 32)
        else:
            # Default size for a handful or pretrained models.
            self.transform = transformfactory(224, 224)

        if self.dataset_name == "cifar10":
            self.native_dimensions = (3, 32, 32)
            self._num_classes = 10
            self.train_set = torchvision.datasets.CIFAR10(
                root=root, train=True, transform=self.transform, download=download
            )
            self.val_set = torchvision.datasets.CIFAR10(
                root=root, train=False, transform=self.transform, download=download
            )
        elif self.dataset_name == "cifar100":
            self.native_dimensions = (3, 32, 32)
            self._num_classes = 100
            self.train_set = torchvision.datasets.CIFAR100(
                root=root, train=True, transform=self.transform, download=download
            )
            self.val_set = torchvision.datasets.CIFAR100(
                root=root, train=False, transform=self.transform, download=download
            )
        elif self.dataset_name == "food101":
            self.native_dimensions = (
                3,
                224,
                224,
            )  # not actually, but it's a common input size
            self._num_classes = 101
            self.train_set = torchvision.datasets.Food101(
                root=root, split="train", transform=self.transform, download=download
            )
            self.val_set = torchvision.datasets.Food101(
                root=root, split="test", transform=self.transform, download=download
            )
        else:
            raise ValueError("Unsupported dataset name: {}".format(self.dataset_name))

        self.distinct_images = len(self.train_set) + len(self.val_set)

    def get_transform(self):
        return self.transform

    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            prefetch_factor=64,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self._resource_man.hw_threads // 4 + 1,
        )

    def get_val_loader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            prefetch_factor=64,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self._resource_man.hw_threads // 4 + 1,
        )

    def __iter__(self):
        self._train_loader_iter = iter(self.get_train_loader())
        return self

    def __next__(self):
        return next(self._train_loader_iter)

    def __len__(self) -> int:
        """Batches in set."""
        return len(self.get_train_loader())

    @property
    def img_channels(self) -> int:
        """Number of channels in the input images."""
        return self.native_dimensions[0]

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def batch_size(self) -> int:
        """Determines batch size of everything downstream"""
        return self._batch_size
