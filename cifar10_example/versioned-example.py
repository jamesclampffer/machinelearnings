"""
Copyright: Jim Clampffer 2025

A simple convolutional neural network that's quick to retrain.

"""

import argparse
import torch
import torch.utils
import torch.nn as nn
import torch.optim  # as opt
import torch.optim.lr_scheduler
import torchvision
import torchvision.datasets
import torchvision.utils


# Define the "shape" of the network
class NaiveNet(torch.nn.Module):
    """A very simple model architecture"""

    def __init__(self):
        """Set up the operators this simple model will use"""
        super(NaiveNet, self).__init__()

        # Convolution layers followed by 2x2->1x1 downsampling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(16)  # 16 chan
        self.activate_fn1 = torch.relu
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second conv and pool, reduce to 8px*8px, 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.norm2 = nn.BatchNorm2d(32)  # 32 chan
        self.activate_fn2 = torch.relu
        self.pool2 = nn.MaxPool2d(2, 2)

        # A third conv layer, same input/output channels and map size
        # TODO: look at pooling and adding channels here
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(32)
        self.activate_fn3 = torch.relu

        # Yet another conv layer
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.norm5 = nn.BatchNorm2d(32)
        self.activate_fn5 = torch.relu
        # self.pool5 = nn.MaxPool2d(2,2)

        # A 4th conv layer
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm2d(32)
        self.activate_fn4 = torch.relu
        self.pool4 = nn.MaxPool2d(2, 2)

        # 8px * 8px * 32 chan -> 10 classes of objects
        self.pre_fc_dropout = nn.Dropout(0.1)
        self.fc_classifier = nn.Linear(3 * 3 * 32, 100)

    def forward(self, imgdata):
        # First convolution + rectification +  pool
        imgdata = self.conv1(imgdata)
        imgdata = self.norm1(imgdata)
        imgdata = self.activate_fn1(imgdata)
        imgdata = self.pool1(imgdata)

        # second convolution + pool
        imgdata = self.conv2(imgdata)
        imgdata = self.norm2(imgdata)
        imgdata = self.activate_fn2(imgdata)
        imgdata = self.pool2(imgdata)

        # third conv layer, no pooling
        imgdata = self.conv3(imgdata)
        imgdata = self.norm3(imgdata)
        imgdata = self.activate_fn3(imgdata)
        res = imgdata

        # When in doubt, add more conv
        imgdata = self.conv5(imgdata)
        imgdata = self.norm5(imgdata)
        imgdata = self.activate_fn5(imgdata)

        # fwd resnet style
        imgdata = imgdata + res

        # 4th conv layer, no pooling
        imgdata = self.conv4(imgdata)
        imgdata = self.norm4(imgdata)
        imgdata = self.activate_fn4(imgdata)
        imgdata = self.pool4(imgdata)

        # Linearize ahead of fully connected layer
        imgdata = imgdata.view(imgdata.size(0), -1)

        # Zero out some weights to reduce redundancy and avoid overfitting
        self.pre_fc_dropout(imgdata)

        # FC layer to determine class of object in img
        clz = self.fc_classifier(imgdata)
        return clz


# Command line params
CPU_NUM = None
PREFETCH_FACTOR = None
EPOCHS = None
BATCH_SIZE = None
INITIAL_LR = None
INITIAL_MOMENTUM = None
STATS_INTERVAL = None
INTEROP_THREADS = None


def main():
    # Speed up training where possible
    torch.set_num_threads(CPU_NUM)
    torch.set_num_interop_threads(INTEROP_THREADS)

    # Use cuda if available, ROCm doesn't support my Radeon 780m
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Augment images used for training
    default_xfm = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomCrop(32, padding=4),  # Crop up to 4 pixels
            torchvision.transforms.RandomHorizontalFlip(p=0.5),  # Flip the image
            # torchvision.transforms.RandomRotation(10),  # wobble side to side
            torchvision.transforms.ColorJitter(
                0.1, 0.1, 0.1, 0.02
            ),  # Fuzz colors and brightness
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Don't bother transforming test set (is this correct?)
    passthrough_xfm = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Download the CIFAR100 dataset, or use local copy if it's there
    training_data = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=default_xfm
    )
    training_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=CPU_NUM,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    # Some images are set aside for validation.
    validation_data = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=passthrough_xfm
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=CPU_NUM,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    # Stand up a model on the compute resource
    model = NaiveNet().to(device)

    # Ignore for now, required for training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=INITIAL_LR, momentum=INITIAL_MOMENTUM
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # The whole training loop
    for epoch in range(EPOCHS):
        """The training loop, run through the whole training set 10 times (epochs)"""
        print("Training epoch {} of {}".format(epoch + 1, EPOCHS))
        model.train()
        loss_acc = 0
        for images, labels in training_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backprop
            loss.backward()

            # recompute learning rate
            optimizer.step()
            scheduler.step()

            loss_acc += loss.item()
        print(
            "Done training epoch {}/{}, loss for epoch: {}, lr: {}".format(
                epoch + 1, EPOCHS, loss_acc, scheduler.get_last_lr()[0]
            )
        )

        # Print acc on an interval, it's expensive to do every epoch
        if True:
            acc = check_acc(model, validation_loader, device)
            print("Epoch acc: {}".format(acc))
        else:
            print("Skipping test check for epoch {}".format(epoch))

    # End of training loop

    # Check accuracy against data the model has not seen
    num = check_acc(model, validation_loader, device)
    print("Overall test accuracy: {}".format(num))


def check_acc(model, loader, device):
    """Run the given set against the model \
    Note: The dataset's transforms will be re-applied"""
    model.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outvec = model(images)
            discard, predicates = torch.max(outvec, 1)
            acc += (predicates == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        return 0.0
    return float(acc) / float(total) * 100.0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simple nn")
    p.add_argument("--cpu_num", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--initial_lr", type=float, default=0.01)
    p.add_argument("--initial_momentum", type=float, default=0.9)
    p.add_argument(
        "--stats_interval",
        type=int,
        default=5,
        help="Calculate accuracy every N epochs",
    )
    args = p.parse_args()

    CPU_NUM = args.cpu_num
    PREFETCH_FACTOR = args.prefetch_factor
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    INITIAL_LR = args.initial_lr
    INITIAL_MOMENTUM = args.initial_momentum
    STATS_INTERVAL = args.stats_interval

    INTEROP_THREADS = int(CPU_NUM / 2) if CPU_NUM > 2 else 1

    main()
