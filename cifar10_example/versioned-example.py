"""
Copyright: Jim Clampffer 2025

A simple convolutional neural network that's quick to retrain.

"""

import torch
import torch.utils
import torch.nn as nn
import torch.optim as opt
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
        self.activate_fn1 = torch.relu
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second conv and pool, reduce to 8px*8px, 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.activate_fn2 = torch.relu
        self.pool2 = nn.MaxPool2d(2, 2)

        # 8px * 8px * 32 chan -> 10 classes of objects
        self.fc_classifier = nn.Linear(1568, 10)

    def forward(self, imgdata):
        # First convolution + rectification +  pool
        imgdata = self.conv1(imgdata)
        imgdata = self.activate_fn1(imgdata)
        imgdata = self.pool1(imgdata)

        # second convolution + pool
        imgdata = self.conv2(imgdata)
        imgdata = self.activate_fn2(imgdata)
        imgdata = self.pool2(imgdata)

        # Linearize ahead of fully connected layer
        imgdata = imgdata.view(imgdata.size(0), -1)

        # FC layer to determine class of object in img
        clz = self.fc_classifier(imgdata)
        return clz


EPOCHS = 10


def main():
    # Speed up training where possible
    torch.set_num_threads(12)
    torch.set_num_interop_threads(2)

    # Use cuda if available, ROCm doesn't support my Radeon 780m
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # img formatting
    default_xfm = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomCrop(32, padding=4),  # Crop up to 4 pixels
            torchvision.transforms.RandomHorizontalFlip(p=0.5),  # Flip the image
            torchvision.transforms.ColorJitter(
                0.2, 0.2, 0.2, 0.02
            ),  # Fuzz colors and brightness
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    passthrough_xfm = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Download the CIFAR10 dataset, or use local copy if it's there
    training_data = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=default_xfm
    )
    training_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Some images are set aside for validation.
    validation_data = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=passthrough_xfm
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Stand up a model on the compute resource
    model = NaiveNet().to(device)

    # Ignore for now, required for training
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # The whole training loop
    for epoch in range(EPOCHS):
        """The training loop, run through the whole training set 10 times (epochs)"""
        print("Training epoch{} of {}".format(epoch + 1, EPOCHS))
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

    # Check accuracy against data the model has not seen
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print("Overall test accuracy: {}".format(100 * correct / float(total)))


if __name__ == "__main__":
    main()
