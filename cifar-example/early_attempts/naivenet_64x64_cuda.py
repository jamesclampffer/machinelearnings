"""
Copyright: Jim Clampffer 2025

A simple convolutional neural network that's quick to retrain.

"""

import argparse
import torch
import torch.cuda.amp
import torch.utils
import torch.nn as nn
import torch.nn.functional
import torch.optim
import torch.optim.lr_scheduler
import torchvision
import torchvision.datasets
import torchvision.transforms


class SimpleResBlock(nn.Module):
    def __init__(self, chan_io):
        super().__init__()
        self.conv1 = nn.Conv2d(chan_io, chan_io, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(chan_io)
        self.activation_fn = nn.ReLU()
        self.conv2 = nn.Conv2d(chan_io, chan_io, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(chan_io)

    def forward(self, x):
        initial = x
        layer1 = self.activation_fn(self.norm1(self.conv1(x)))
        layer2 = self.norm2(self.conv2(layer1))
        layer2 += initial
        return self.activation_fn(layer2)


class SimpleBottleneck(nn.Module):
    def __init__(self, in_chan, out_chan, neck_chan):
        super().__init__()
        # expand channels TODO: sequential.Compose these
        self.explode = nn.Conv2d(in_chan, neck_chan, kernel_size=1)
        self.explode_norm = nn.BatchNorm2d(neck_chan)
        self.explode_activate = nn.ReLU()

        # depthwise conv through expanded channels
        self.core = nn.Conv2d(neck_chan, neck_chan, kernel_size=3, padding=1)
        self.core_norm = nn.BatchNorm2d(neck_chan)
        self.core_activate = nn.ReLU()

        # reduce channel count
        self.implode = nn.Conv2d(neck_chan, out_chan, kernel_size=1)
        self.implode_norm = nn.BatchNorm2d(out_chan)
        self.implode_activate = nn.ReLU()

    def forward(self, x):
        x = self.explode(x)
        x = self.explode_norm(x)
        x = self.explode_activate(x)
        x = self.core(x)
        x = self.core_norm(x)
        x = self.core_activate(x)
        x = self.implode(x)
        x = self.implode_norm(x)
        x = self.implode_activate(x)
        return x


class SimpleConvBlock(nn.Module):
    """Stop repeating conv->norm->relu->pool"""

    def __init__(self, in_chan, out_chan, enable_pool, activation_fn=torch.relu):
        super(SimpleConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_chan)
        self.activation_fn = activation_fn

        self.enable_pool = enable_pool
        if enable_pool:
            self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.05)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation_fn(x)
        if self.enable_pool:
            x = self.pool(x)
        x = self.drop(x)
        return x


# Using CIFAR10/CIFAR100 scaled up by some integer
IMG_X = 64
IMG_Y = 64


# Define the "shape" of the network
class NaiveNet(torch.nn.Module):
    """A very simple model architecture"""

    def __init__(self):
        """Set up the operators this simple model will use"""
        super().__init__()

        # Convolution layers followed by 2x2->1x1 downsampling
        self.neck1 = SimpleBottleneck(3, 32, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.res1 = SimpleResBlock(32)

        # Second conv and pool, reduce to 8px*8px, 32 channels
        self.convblock2 = SimpleConvBlock(32, 64, True)
        self.res2 = SimpleResBlock(64)
        # A third conv layer, same input/output channels and map size
        self.convblock3 = SimpleConvBlock(64, 64, False)
        self.res3 = SimpleResBlock(64)
        self.neck3 = SimpleBottleneck(64,64,128)

        # Yet another conv layer
        self.convblock4 = SimpleConvBlock(64, 64, False)

        # A 4th conv layer
        self.convblock5 = SimpleConvBlock(64, 32, True)

        # 8px * 8px * 32 chan -> 10 classes of objects
        self.pre_fc_dropout = nn.Dropout(0.1)
        self.fc_classifier = None

    def forward(self, imgdata):
        # First convolution + rectification +  pool
        imgdata = self.neck1(imgdata)
        imgdata = self.pool1(imgdata)
        imgdata = self.res1(imgdata)

        # second convolution + pool
        imgdata = self.convblock2(imgdata)
        imgdata = self.res2(imgdata)

        # third conv layer, no pool
        imgdata = self.convblock3(imgdata)
        imgdata = self.res3(imgdata)
        imgdata = self.neck3(imgdata)

        # When in doubt, add more conv
        imgdata = self.convblock4(imgdata)

        # 4th conv layer, no pooling
        imgdata = self.convblock5(imgdata)

        # print("Feature shape before flattening:", imgdata.shape)

        imgdata = self.pre_fc_dropout(imgdata)
        imgdata = torch.nn.functional.adaptive_avg_pool2d(imgdata, 1)
        imgdata = imgdata.view(imgdata.size(0), -1)

        if self.fc_classifier is None:
            self.fc_classifier = nn.Linear(imgdata.size(1), 100).to(imgdata.device)

        return self.fc_classifier(imgdata)


# Command line params
CPU_NUM = None
PREFETCH_FACTOR = None
EPOCHS = None
BATCH_SIZE = None
INITIAL_LR = None
INITIAL_MOMENTUM = None
STATS_INTERVAL = None
INTEROP_THREADS = None
SKIP_LAYER = None


def main():
    # Speed up training where possible
    torch.set_num_threads(CPU_NUM)
    torch.set_num_interop_threads(INTEROP_THREADS)

    # Use cuda if available, ROCm doesn't support my Radeon 780m
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    enable_cuda_amp = device_type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=enable_cuda_amp)

    # Augment images used for training
    default_xfm = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomCrop(IMG_X, padding=4),  # Crop up to 4 pixels
            torchvision.transforms.RandomHorizontalFlip(p=0.5),  # Flip the image
            torchvision.transforms.ColorJitter(
                0.1, 0.1, 0.1, 0.02
            ),  # Fuzz colors and brightness
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Don't bother transforming test set (is this correct?)
    passthrough_xfm = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((64, 64)),
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
        pin_memory=True,
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
        pin_memory=True,
    )

    # Stand up a model on the compute resource
    model = NaiveNet().to(device)

    # Ignore for now, required for training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=INITIAL_LR, momentum=INITIAL_MOMENTUM, weight_decay=1e-4
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
            # The syntax here changes with versions
            with torch.cuda.amp.autocast(enabled=enable_cuda_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_acc += loss.item()
        scheduler.step()

        print(
            "Done training epoch {}/{}, loss for epoch: {}, lr: {}".format(
                epoch + 1, EPOCHS, loss_acc, scheduler.get_last_lr()[0]
            )
        )

        # Print acc on an interval, it's expensive to do every epoch
        if (epoch + 1) % STATS_INTERVAL == 0:
            acc = check_acc(model, validation_loader, device)
            print("Epoch acc: {}".format(acc))
        else:
            print("Skipping test check for epoch {}".format(epoch))

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
    p.add_argument(
        "--cpu_num", type=int, default=6, help="Cores to dedicate to process"
    )
    p.add_argument("--prefetch_factor", type=int, default=12)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--initial_lr", type=float, default=0.01)
    p.add_argument("--initial_momentum", type=float, default=0.9)
    p.add_argument(
        "--stats_interval",
        type=int,
        default=5,
        help="Calculate accuracy every N epochs",
    ),
    p.add_argument("--skip_layer", type=bool, default=False, help="currently ignored")
    args = p.parse_args()

    CPU_NUM = args.cpu_num
    PREFETCH_FACTOR = args.prefetch_factor
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    INITIAL_LR = args.initial_lr
    INITIAL_MOMENTUM = args.initial_momentum
    STATS_INTERVAL = args.stats_interval
    SKIP_LAYER = args.skip_layer

    INTEROP_THREADS = int(CPU_NUM / 1.5) if CPU_NUM > 2 else 1

    main()
