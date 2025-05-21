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

# Command line params
CPU_NUM = None
PREFETCH_FACTOR = None
EPOCHS = None
BATCH_SIZE = None
INITIAL_LR = None
INITIAL_MOMENTUM = None
STATS_INTERVAL = None
INTEROP_THREADS = None


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


class SimpleResBlock(nn.Module):
    """Plain conv with residual block. Some of the other blocks defined below handle residuals as well, but do more things"""

    def __init__(self, chan_io, activation_fn=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(chan_io, chan_io, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(chan_io)
        self.activation_fn = activation_fn()
        self.conv2 = nn.Conv2d(chan_io, chan_io, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(chan_io)

    def forward(self, x):
        initial = x
        layer1 = self.activation_fn(self.norm1(self.conv1(x)))
        layer2 = self.norm2(self.conv2(layer1))
        layer2 += initial
        return self.activation_fn(layer2)


class SimpleBottleneck(nn.Module):
    def __init__(self, in_chan, out_chan, neck_chan, activation_fn=nn.ReLU):
        super().__init__()
        # expand channels TODO: sequential.Compose these
        self.explode = nn.Conv2d(in_chan, neck_chan, kernel_size=1)
        self.explode_norm = nn.BatchNorm2d(neck_chan)
        self.explode_activate = activation_fn()

        # depthwise conv through expanded channels
        self.core = nn.Conv2d(neck_chan, neck_chan, kernel_size=3, padding=1)
        self.core_norm = nn.BatchNorm2d(neck_chan)
        self.core_activate = activation_fn()

        # reduce channel count
        self.implode = nn.Conv2d(neck_chan, out_chan, kernel_size=1)
        self.implode_norm = nn.BatchNorm2d(out_chan)
        self.implode_activate = activation_fn()

    def forward(self, x):
        # Add a skiplayer option here?
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


class DepthwiseSeperableBottleneck(nn.Module):
    """Resnet50 style bottleneck with Xception's depthwise conv trick"""

    def __init__(
        self, chan_in, chan_out, chan_core, activation_fn=torch.relu, fwd_res=True
    ):
        super().__init__()
        self.use_residual = chan_in == chan_out and fwd_res
        self.activation_fn = activation_fn

        self.explode = nn.Conv2d(chan_in, chan_core, kernel_size=1)
        self.norm_in = nn.BatchNorm2d(chan_core)

        # Channel count should be high here, so do a depthwise conv to keep things a little faster
        self.coreconv = nn.Conv2d(
            chan_core, chan_core, kernel_size=3, stride=1, padding=1, groups=chan_core
        )
        self.corenorm = nn.BatchNorm2d(chan_core)

        self.implode = nn.Conv2d(chan_core, chan_out, kernel_size=1)
        self.norm_out = nn.BatchNorm2d(chan_out)

    def forward(self, x):
        out = self.activation_fn(self.norm_in(self.explode(x)))
        out = self.activation_fn(self.corenorm(self.coreconv(out)))
        out = self.norm_out(self.implode(out))
        if self.use_residual:
            out = out + x
        return torch.relu(out)


# Using CIFAR10/CIFAR100 scaled up by some integer
IMG_X = 128
IMG_Y = 128


# Define the "shape" of the network
class NaiveNet(torch.nn.Module):
    """
    A very simple cobbled together architecture.
    Optimized for CIFAR100, or ar least 32x32px input

    My intent to improve training by upscaleing low-res 32x32 cifar
    in order to augment training data. At 32x32 xforms other than
    pixel translation e.g. rotation, shear will be extremely lossy.
    This approach scales it up to 128x128 (int multiple of 32) such
    that it's still small and fast. Spatial granularity allows more
    training augmentation transforms. It'd be interesting to see how
    well this generalizes to larger, or less upscaled, input during
    inference after being trained on tiny images.

    The downside of this approach is the extra compute required to
    deal with images > 32x32. I did not want to increase training
    time either.

    Themes from resnet, mobilenet, searching, and earlier attepts.
    - forwarding / residual blocks
    - depthwise seperable conv on maps with many channels
    - sprinking conv layers around generally helps, esp if already
      want to change channel count.

    Data flow, roughly
        Bottleneck>MaxPool2x2->Conv
        Repeated conv+res block pairs while fattening channels
        DepthwiseDeperableBottleneck->less params/runtime
        Repeated conv+res block pairs while flatting channels
        Conv->Flatten->FC
    """

    def __init__(self):
        """Set up the operators this simple model will use"""
        super().__init__()

        # expand channels without a ton of compute
        self.neck1 = SimpleBottleneck(3, 16, 24)
        # downsample, this was optimized for 32x32 cifar100
        self.pool1 = nn.MaxPool2d(2, 2)
        # initial feature detection, hang on to spatial data
        self.res1 = SimpleResBlock(16)

        # further fan out channels, with another 2x2-> pool
        self.convblock2 = SimpleConvBlock(16, 64, True)
        # keep identifying smaller features, but keep spatial info
        self.res2 = SimpleResBlock(64)

        # TBD: strided conv rather than pooling
        self.convblock3 = SimpleConvBlock(64, 64, True)
        self.res3 = SimpleResBlock(64)
        self.neck3 = DepthwiseSeperableBottleneck(64, 96, 192)

        # Conv layer to narrow channels.
        self.convblock4 = SimpleConvBlock(96, 64, False)
        self.res4 = SimpleResBlock(64)

        # Further narrow
        self.convblock5 = SimpleConvBlock(64, 32, True)
        self.pre_fc_dropout = nn.Dropout(0.1)

        # Gross. Is there not a better way to do this?
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_Y, IMG_X)
            dummy_out = self.fwd_logic(dummy)
            flat_dim = dummy_out.view(1, -1).shape[1]
            self.fc_classifier = nn.Linear(flat_dim, 100)

    def fwd_logic(self, imgdata):
        """Encapsulate forward() logic so it can be referenced in init"""
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
        imgdata = self.res4(imgdata)

        # 4th conv layer, no pooling
        imgdata = self.convblock5(imgdata)

        imgdata = self.pre_fc_dropout(imgdata)
        imgdata = torch.nn.functional.adaptive_avg_pool2d(imgdata, 1)
        return imgdata

    def forward(self, imgdata):
        out = self.fwd_logic(imgdata)
        out = out.view(imgdata.size(0), -1)
        return self.fc_classifier(out)


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
            torchvision.transforms.Resize((IMG_X, IMG_Y)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomCrop(IMG_X, padding=4),  # Crop up to 4 pixels
            torchvision.transforms.RandomHorizontalFlip(p=0.5),  # Flip the image
            # todo: add RandomAffine when dims > 128x128
            torchvision.transforms.ColorJitter(
                0.1, 0.1, 0.1, 0.02
            ),  # Fuzz colors and brightness
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Don't bother transforming test set (is this correct?)
    passthrough_xfm = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((IMG_X, IMG_Y)),
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
    print("Loaded model")

    par_total = sum(p.numel() for p in model.parameters())
    par_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "Arch data:\n\ttotal params: {}\n\ttrainable params: {}".format(
            par_total, par_train
        )
    )

    # Ignore for now, required for training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=INITIAL_LR, momentum=INITIAL_MOMENTUM, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # TODO: refactor now that I'm more familiar with the library
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

    # Assign options to globals..
    CPU_NUM = args.cpu_num
    PREFETCH_FACTOR = args.prefetch_factor
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    INITIAL_LR = args.initial_lr
    INITIAL_MOMENTUM = args.initial_momentum
    STATS_INTERVAL = args.stats_interval

    # Derived constants
    INTEROP_THREADS = int(CPU_NUM / 1.5) if CPU_NUM > 2 else 1

    main()
