"""
Copyright: Jim Clampffer 2025

Proof of concept for a hacked together model to train on CIFAR-10/100 images.

This is a script, not a reusable module.
"""

import argparse
import os
import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional
import torch.optim
import torch.optim.lr_scheduler
import torchvision
import torchvision.datasets
import torchvision.transforms
import torchinfo
import warnings

# Version handling
try:
    # Deprecated, warning is suppressed
    import torch.cuda.amp as amp
except ImportError:
    # The new way, which does not exist on older versions
    import torch.amp as amp

# Program parameter priority: command line >> environment vars >> default
# This must happen before anything else has a chance to be instantiated.
CPU_COUNT = os.cpu_count()
PREFETCH_FACTOR = os.getenv("PREFETCH_FACTOR")
EPOCHS = os.getenv("EPOCH_COUT")
BATCH_SIZE = os.getenv("BATCH_SIZE")
INITIAL_LR = os.getenv("INITIAL_LR")
INITIAL_MOMENTUM = os.getenv("INITIAL_MOMENTUM")
STATS_INTERVAL = os.getenv("STATS_INTERVAL")
INTEROP_THREADS = os.getenv("INTEROP_THREADS")
SCALE_CHAN = os.getenv("SCALE_CHAN")

# Upscaled img size. Some integer multiple of 32. Extra pixels give training
# data augmentation transforms room to interpolate using a more granular
# representation; try a rotation at 32x32 and you're gonna have a bad time.
IMG_X = 128
IMG_Y = 128


def scale_chan(ch):
    """
    Scale nominal channel count to experiment with model size
    """
    global SCALE_CHAN
    return int(ch / SCALE_CHAN) if ch > 8 else ch


class Conv2dNorm2d(torch.nn.Module):
    """
    Almost all 2d convs are followed by 2d batchnorm in this project, so
    codify that pattern. Any benefit vs. factory function that directly
    returns a Sequential?
    """

    __slots__ = "seq"

    def __init__(self, chan_in, chan_out, norm_momentum=0.05):
        super().__init__()
        self.seq = torch.nn.Sequential(
            nn.Conv2d(chan_in, chan_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(chan_out, momentum=norm_momentum),
        )

    def forward(self, fmap):
        return self.seq(fmap)


class SimpleConvBlock(nn.Module):
    """
    Conv with optional skip-layer
    """

    __slots__ = "seq", "use_residual"

    def __init__(
        self,
        chan_in,
        chan_out,
        activation_fn=nn.SiLU,
        kernel_size=3,
        padding=1,
        stride=1,
    ):
        super().__init__()
        self.use_residual = chan_in == chan_out and stride == 1

        self.seq = nn.Sequential(
            nn.Conv2d(
                scale_chan(chan_in),
                scale_chan(chan_out),
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm2d(scale_chan(chan_out), momentum=0.05),
            activation_fn(),
            nn.Dropout2d(0.05),
        )

    def forward(self, fmap):
        fmap_out = self.seq(fmap)

        if self.use_residual:
            fmap_out += fmap
        return fmap_out


class SimpleResidualBlock(nn.Module):
    """
    Resnet-style residual block. Some other blocks also re-add the
    residual but also do bonus things.
    """

    __slots__ = "seq", "activation_fn"

    def __init__(self, chan_io, activation_fn=nn.SiLU):
        super().__init__()
        self.activation_fn = activation_fn()

        ch = scale_chan(chan_io)

        self.seq = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch, momentum=0.05),
            self.activation_fn,
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch, momentum=0.05),
        )

    def forward(self, fmap_in):
        fmap_out = self.seq(fmap_in)
        fmap_out = fmap_out + fmap_in  # Add residual
        return self.activation_fn(fmap_out)


class SqueezeExciteBlock(nn.Module):
    """
    Aim attention at the most active channels for an input
    """

    __slots__ = "seq"

    def __init__(self, chan, ratio=4):
        super().__init__()
        chan = scale_chan(chan)
        self.seq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(chan, chan // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(chan // ratio, chan, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.seq(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class SimpleBottleneckBlock(nn.Module):
    """
    Take an input with a low number of channels output one with more.

    Convolution pass expands the channel count significantly, followed by
    another convolution that maintains dimensions while expanded. Finally, a
    last convolution partially reduces the number of channels.
    """

    __slots__ = "explode", "core", "implode", "squeeze"

    def __init__(self, chan_in, chan_out, neck_chan, activation_fn=nn.SiLU, stride=1):
        super().__init__()
        explode_activate = activation_fn
        core_activate = activation_fn
        implode_activate = activation_fn

        scaled_chan_in = scale_chan(chan_in)
        ch_out = scale_chan(chan_out)
        ch_neck = scale_chan(neck_chan)

        self.explode = torch.nn.Sequential(
            nn.Conv2d(scaled_chan_in, ch_neck, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_neck, momentum=0.05),
            explode_activate(),
        )

        self.core = torch.nn.Sequential(
            # todo: try a quickconv block
            nn.Conv2d(
                ch_neck, ch_neck, kernel_size=3, padding=1, bias=False, stride=stride
            ),
            nn.BatchNorm2d(ch_neck, momentum=0.05),
            core_activate(),
        )

        self.implode = torch.nn.Sequential(
            nn.Conv2d(ch_neck, ch_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_out, momentum=0.05),
            implode_activate(),
        )

        self.squeeze = SqueezeExciteBlock(ch_out)

    def forward(self, fmap):
        fmap = self.explode(fmap)
        fmap = self.core(fmap)
        fmap = self.implode(fmap)
        return self.squeeze(fmap)


class DepthwiseSeparableBottleneck(nn.Module):
    """
    Resnet style bottleneck with Xception's depthwise separable conv strength
    reduction trick.
    """

    __slots__ = "seq", "chan_in", "chan_out", "use_residual"

    def __init__(
        self,
        chan_in,
        chan_out,
        chan_core,
        activation_fn=nn.SiLU,
        fwd_res=True,
        stride=1,
    ):
        super().__init__()
        self.use_residual = chan_in == chan_out and fwd_res
        scaled_chan_in = scale_chan(chan_in)
        ch_out = scale_chan(chan_out)
        ch_core = scale_chan(chan_core)

        self.seq = nn.Sequential(
            nn.Conv2d(scaled_chan_in, ch_core, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_core, momentum=0.05),
            activation_fn(),
            nn.Conv2d(
                ch_core,
                ch_core,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=ch_core,
                bias=False,
            ),
            nn.BatchNorm2d(ch_core, momentum=0.05),
            activation_fn(),
            nn.Conv2d(ch_core, ch_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_out, momentum=0.05),
            activation_fn(),
            SqueezeExciteBlock(ch_out),
        )

    def forward(self, fmap):
        fmap_out = self.seq(fmap)
        if self.use_residual:
            fmap_out = fmap_out + fmap
        return fmap_out


class QuickConvBlock(nn.Module):
    """
    Channelwise conv followed by pointwise conv. Same goals and tradeoffs
    as the DepthwiseSeparableBottleneck efficiency trick, but without the
    bottleneck stuff.
    """

    __slots__ = "seq", "use_residual"

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size=3,
        stride=1,
        activation_fn=nn.SiLU,
    ):
        super().__init__()
        self.use_residual = chan_in == chan_out and stride == 1
        scaled_chan_in = scale_chan(chan_in)
        scaled_chan_out = scale_chan(chan_out)

        self.seq = nn.Sequential(
            # Use groups = input channels to get per-channel conv
            nn.Conv2d(
                scaled_chan_in,
                scaled_chan_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=scaled_chan_in,
                bias=False,
            ),
            activation_fn(),
            nn.Conv2d(
                scaled_chan_in,
                scaled_chan_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(scaled_chan_out, momentum=0.05),
            activation_fn(),
        )

    def forward(self, fmap):
        fmap_out = self.seq(fmap)
        if self.use_residual:
            fmap_out = fmap_out + fmap
        return fmap_out


class NaiveNet(torch.nn.Module):
    """
    A very simple cobbled together architecture.
    Optimized for CIFAR-10/100, or at least 32x32px input

    My intent is to improve training by upscaling low-res 32x32 CIFAR. At
    32x32, affine transforms and other distortions destroy significant
    information due to coarse interpolation. Here, inputs are taken as
    128x128 (without interpolation from 32x32), providing better spatial
    granularity. Intuitively, this should also act somewhat like cutout
    augmentation, as the blocky upscaled image has "gaps" around the object
    periphery.

    The downside of this approach is the extra compute required to process
    images larger than 32x32.

    Themes from resnet, mobilenet, searching, and earlier attempts.
    - forwarding / residual blocks
    - depthwise separable conv on large feature maps to reduce overhead
    - Squeeze-Excite blocks for channel attention
    """

    __slots__ = "frontend", "main_pipeline", "backend"

    def __init__(self):
        """
        Set up the operators this simple model will use
        """
        super().__init__()

        # Bottleneck to increase the number of channels, then run strided
        # convs to reduce feature map dims.
        self.frontend = nn.Sequential(
            SimpleBottleneckBlock(3, 16, 36, activation_fn=nn.SiLU, stride=2),
            SimpleResidualBlock(16, activation_fn=nn.SiLU),
            SimpleConvBlock(
                16,
                64,
                activation_fn=nn.SiLU,
                stride=2,
            ),
        )

        # There should still be plenty of spatial information due to earlier
        # residual flow. Run the feature map through several convolutions,
        # then use a bottleneck to add more channels. Channel and spatial
        # reduction happens next.
        self.main_pipeline = nn.Sequential(
            SimpleResidualBlock(64),
            QuickConvBlock(64, 64, stride=1),
            SimpleResidualBlock(64),
            DepthwiseSeparableBottleneck(64, 96, 192),
        )

        # Spatial downsampling + channel reduction prior to FC. The final
        # downsampling is left to adaptive pooling later on.
        self.backend = nn.Sequential(
            QuickConvBlock(96, 64),
            SimpleResidualBlock(64),
            SimpleConvBlock(64, 32, activation_fn=nn.SiLU, stride=2),
            nn.Dropout(0.1),
        )

        # There has to be a smarter way to calculate this dim without running
        # a fake forward pass, right?
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_Y, IMG_X)
            dummy_out = self.fwd_logic(dummy)
            # Check the output shape to see how wide to make the FC classifier
            flat_dim = dummy_out.view(1, -1).shape[1]
            self.fc_classifier = nn.Linear(flat_dim, 100)

    def fwd_logic(self, imgdata):
        """
        Indirection to forward() logic so it can be referenced in init
        """
        # Groups for different parts of NN
        imgdata = self.frontend(imgdata)
        imgdata = self.main_pipeline(imgdata)
        imgdata = self.backend(imgdata)

        # Not sure if I want this and FC to be grouped
        imgdata = torch.nn.functional.adaptive_avg_pool2d(imgdata, 1)
        return imgdata

    def forward(self, imgdata):
        out = self.fwd_logic(imgdata)
        out = out.view(imgdata.size(0), -1)
        return self.fc_classifier(out)


def check_img_upscale() -> None:
    """
    Make sure scale is a multiple of 32.

    Avoiding interpolation while upscaling small CIFAR-X images is critical to
    avoid losing information. One pixel must cleanly map to n*n pixels.
    """
    assert IMG_X % 32 == 0, "X scaling factor must be an integer. Currently {}".format(
        IMG_X / 32.0
    )
    assert IMG_Y % 32 == 0, "Y scaling factor must be an integer. Currently {}".format(
        IMG_Y / 32.0
    )


def get_training_transform() -> torchvision.transforms.Compose:
    """
    Transform to apply prior to use in training. Includes augmentation
    """
    check_img_upscale()

    # Look into using torch.jit on this. Or at least the bulk of it that
    # doesn't use PIL.
    return torchvision.transforms.Compose(
        [
            # Scale up such that 1px -> n*n px without interpolation.
            torchvision.transforms.Resize((IMG_X, IMG_Y)),
            # Crop to reduce reliance on features at the periphery of the image.
            torchvision.transforms.RandomCrop(IMG_X, padding=4),
            # Horz flip since the source data consists of things that can
            # rotate on the ground. A vert flip would make a lot less sense -
            # up/down encodes useful info.
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            # Less expensive than warping, also get rotation as a package deal
            torchvision.transforms.RandomAffine(degrees=10, shear=5),
            # Fuzz colors, brightness etc.
            torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def get_validation_transform() -> torchvision.transforms.Compose:
    """
    Transform to apply to validation images. Just resize.
    """
    check_img_upscale()
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((IMG_X, IMG_Y)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


# todo: parameterize to toggle between CIFAR-10 and CIFAR-100
def get_data_loaders() -> (
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
):
    """
    Return a loader for training and another for validation
    """
    make_loader = lambda dset, shuff: torch.utils.data.DataLoader(
        dset,
        batch_size=BATCH_SIZE,
        shuffle=shuff,
        num_workers=CPU_COUNT,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=torch.cuda.is_available(),
    )

    # Download the CIFAR100 dataset if a local copy isn't handy
    training_data = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=get_training_transform()
    )

    # Some images are set aside for validation.
    validation_data = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=get_validation_transform()
    )

    training_loader = make_loader(training_data, True)
    validation_loader = make_loader(validation_data, False)

    return (training_loader, validation_loader)


def training_loop(
    executor,
    model: nn.Module,
    epoch_count: int,
    loss_fn,
    loaders,
    scheduler,
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    Running training loop.
    """
    training_loader, validation_loader = loaders

    # Determine if it's safe to use fp16 instead of fp32 for certain tensor
    # ops on a per-batch basis. ML accelerators spend most of their fungible
    # die area on half-precision tensor units at the (relative) cost of wider
    # type throughput.
    float_scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(epoch_count):
        print("Training epoch {} of {}".format(epoch + 1, epoch_count))
        model.train()
        loss_acc = 0
        for images, labels in training_loader:
            images, labels = images.to(executor), labels.to(executor)
            optimizer.zero_grad()

            # Operate on fp16 tensors, if possible
            with amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            # Then step in a type width-aware manner
            float_scaler.scale(loss).backward()
            float_scaler.step(optimizer)
            float_scaler.update()

            # OneCycleLR needs batch-level stepping
            scheduler.step()

            loss_acc += loss.item()

        print(
            "Done training epoch {}/{}, loss for epoch: {}, lr: {}".format(
                epoch + 1, epoch_count, loss_acc, scheduler.get_last_lr()[0]
            )
        )

        # Print on interval, it's expensive to calculate using the full test
        # set on every epoch.
        if (epoch + 1) % STATS_INTERVAL == 0 and epoch != 0:
            acc = check_acc(model, validation_loader, executor)
            print("Epoch {} acc: {}".format(epoch, acc))


def main() -> None:
    # Override if running other workloads
    torch.set_num_threads(CPU_COUNT)
    torch.set_num_interop_threads(INTEROP_THREADS)

    # Use cuda if available. ROCm doesn't support my Radeon 780m :(
    device_type = "cpu"
    if torch.cuda.is_available():
        device_type = "cuda"

    compute_device = torch.device(device_type)

    # dataset loaders
    training_loader, validation_loader = get_data_loaders()

    # Stand up a model on the compute resource
    model = NaiveNet().to(compute_device)
    print(
        torchinfo.summary(
            model,
            input_size=(1, 3, 128, 128),
            col_names=("input_size", "output_size", "num_params"),
        )
    )

    # Experiment with other loss functions. Parameterize label smoothing and
    # try sweeping it along with other params.
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=INITIAL_LR, momentum=INITIAL_MOMENTUM, weight_decay=1e-4
    )

    # This handily beat other schedulers I tried, but should be revisited
    # during hyperparam tuning.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=INITIAL_LR * 5,
        steps_per_epoch=len(training_loader),
        epochs=EPOCHS,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4,
        anneal_strategy="cos",
    )

    # train the model
    training_loop(
        compute_device,
        model,
        EPOCHS,
        loss_fn,
        (training_loader, validation_loader),
        scheduler,
        optimizer,
    )

    # Final full accuracy check.
    finalacc = check_acc(model, validation_loader, compute_device)
    print("Overall validation test accuracy: {}".format(finalacc))

    fname = "model.pth"
    torch.save(model.state_dict(), fname)
    print("saved {}".format(fname))

def check_acc(model, loader, compute_device) -> float:
    """
    Run the given set against the model
    """
    model.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(compute_device), labels.to(compute_device)
            outvec = model(images)
            discard, predicates = torch.max(outvec, 1)
            acc += (predicates == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        return 0.0
    return float(acc) / float(total) * 100.0


if __name__ == "__main__":
    # One cloud vendor I use starts with an older PyTorch version—old enough
    # to not support the torch.amp module. Implementing the suggested fix
    # won't find the package. Upgrade time incurs costs I'd rather avoid.
    warnings.filterwarnings(
        action="ignore",
        category=FutureWarning,
        message=".*torch[.]cuda[.]amp[.].*is deprecated.*",
    )

    p = argparse.ArgumentParser(description="Simple nn training loop")
    p.add_argument("--cpu_count", type=int, default=CPU_COUNT, help="CPU cores to use.")
    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=(PREFETCH_FACTOR if PREFETCH_FACTOR != None else 12),
        help="Lower to reduce memory consumption, raise if gpu is starved",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=(EPOCHS if EPOCHS != None else 20),
        help="Training epochs to execute",
    )
    p.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE if BATCH_SIZE != None else 128
    )
    p.add_argument(
        "--initial_lr",
        type=float,
        default=(INITIAL_LR if INITIAL_LR != None else 0.01),
        help="Initial learning rate",
    )
    p.add_argument(
        "--initial_momentum",
        type=float,
        default=INITIAL_MOMENTUM if INITIAL_MOMENTUM != None else 0.9,
        help="Not applicable to all optimizers",
    )
    p.add_argument(
        "--stats_interval",
        type=int,
        default=STATS_INTERVAL if STATS_INTERVAL != None else 5,
        help="Calculate accuracy every N epochs",
    ),
    p.add_argument(
        "--scale_chan",
        type=float,
        default=SCALE_CHAN if SCALE_CHAN != None else 1.0,
        help="Multiplier for channel count",
    )
    args = p.parse_args()

    # Assign options to globals..
    CPU_COUNT = args.cpu_count
    PREFETCH_FACTOR = args.prefetch_factor
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    INITIAL_LR = args.initial_lr
    INITIAL_MOMENTUM = args.initial_momentum
    STATS_INTERVAL = args.stats_interval
    SCALE_CHAN = args.scale_chan

    # Derived constants
    INTEROP_THREADS = int(CPU_COUNT / 1.5) if CPU_COUNT > 2 else 1

    main()
else:
    # pytorch uses this as a module when multiprocessing is enabled
    pass
