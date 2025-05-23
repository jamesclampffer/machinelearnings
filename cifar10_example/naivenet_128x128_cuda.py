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

# Command line params for visibility here. Set in main()
CPU_NUM = None
PREFETCH_FACTOR = None
EPOCHS = None
BATCH_SIZE = None
INITIAL_LR = None
INITIAL_MOMENTUM = None
STATS_INTERVAL = None
INTEROP_THREADS = None

# Upscaled img size. Some integer multiple of 32. Extra pixels give training
# data augmentation transforms room to interpolate using a more granular
# representation; try a rotation at 32x32 and you're gonna have a bad time.
IMG_X = 128
IMG_Y = 128


class SimpleConvBlock(nn.Module):
    """Conv with optional skip-layer"""

    def __init__(
        self,
        in_chan,
        out_chan,
        enable_pool,
        activation_fn=nn.functional.silu,
        kernel_size=3,
        padding=1,
        stride=1,
    ):
        super(SimpleConvBlock, self).__init__()
        self.activation_fn = activation_fn

        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=kernel_size, padding=padding, stride=stride
        )
        self.norm = nn.BatchNorm2d(out_chan, momentum=0.05)

        self.enable_pool = enable_pool
        if enable_pool:
            self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(0.05)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation_fn(x)
        if self.enable_pool:
            x = self.pool(x)
        x = self.drop(x)
        return x


class SimpleResidualBlock(nn.Module):
    """Plain conv with residual block. Some of the other blocks defined below handle residuals as well, but do more things"""

    def __init__(self, chan_io, activation_fn=nn.functional.silu):
        super().__init__()
        self.activation_fn = activation_fn
        self.conv1 = nn.Conv2d(chan_io, chan_io, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(chan_io, momentum=0.05)
        self.conv2 = nn.Conv2d(chan_io, chan_io, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(chan_io, momentum=0.05)

    def forward(self, x):
        initial = x
        layer1 = self.activation_fn(self.norm1(self.conv1(x)))
        layer2 = self.norm2(self.conv2(layer1))
        layer2 += initial
        return self.activation_fn(layer2)

class SqueezeExciteBlock(nn.Module):
    """Aim attention at the most active channels for an input"""
    def __init__(self, chan, ratio=4):
        super().__init__()
        self.proc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(chan, chan//ratio, bias=False), nn.ReLU(inplace=True),
            nn.Linear(chan//ratio, chan, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.proc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class SimpleBottleneckBlock(nn.Module):
    def __init__(self, in_chan, out_chan, neck_chan, activation_fn=nn.functional.silu, stride=1):
        super().__init__()
        self.explode_activate = activation_fn
        self.core_activate = activation_fn
        self.implode_activate = activation_fn

        # expand channels TODO: sequential.Compose these
        self.explode = nn.Conv2d(in_chan, neck_chan, kernel_size=1, bias=False)
        self.explode_norm = nn.BatchNorm2d(neck_chan, momentum=0.05)
        # depthwise conv through expanded channels
        self.core = nn.Conv2d(neck_chan, neck_chan, kernel_size=3, padding=1, bias=False, stride=stride)
        self.core_norm = nn.BatchNorm2d(neck_chan, momentum=0.05)
        # reduce channel count
        self.implode = nn.Conv2d(neck_chan, out_chan, kernel_size=1, bias=False)
        self.implode_norm = nn.BatchNorm2d(out_chan, momentum=0.05)
        self.chan_squeeze = SqueezeExciteBlock(out_chan)

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
        x = self.chan_squeeze(x)
        return x


class DepthwiseSeperableBottleneck(nn.Module):
    """Resnet50 style bottleneck with Xception's depthwise conv trick"""

    def __init__(
        self, chan_in, chan_out, chan_core, activation_fn=nn.functional.silu, fwd_res=True, stride=1
    ):
        super().__init__()
        self.use_residual = chan_in == chan_out and fwd_res
        self.activation_fn = activation_fn

        self.explode = nn.Conv2d(chan_in, chan_core, kernel_size=1, bias=False)
        self.norm_in = nn.BatchNorm2d(chan_core, momentum=0.05)

        # Do a depthwise conv to keep things a little faster
        self.coreconv = nn.Conv2d(
            chan_core, chan_core, kernel_size=3, stride=stride, padding=1, groups=chan_core, bias=False
        )
        self.corenorm = nn.BatchNorm2d(chan_core, momentum=0.05)

        self.implode = nn.Conv2d(chan_core, chan_out, kernel_size=1, bias=False)
        self.norm_out = nn.BatchNorm2d(chan_out, momentum=0.05)
        self.chan_squeeze = SqueezeExciteBlock(chan_out)

    def forward(self, x):

        out = self.activation_fn(self.norm_in(self.explode(x)))
        out = self.activation_fn(self.corenorm(self.coreconv(out)))
        out = self.activation_fn(self.norm_out(self.implode(out)))
        out = self.chan_squeeze(out)
        if self.use_residual:
            out = out + x
        return out


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
        self.neck1 = SimpleBottleneckBlock(3, 16, 36, activation_fn=nn.functional.silu, stride=2)

        # initial feature detection, hang on to spatial data
        self.res1 = SimpleResidualBlock(16, activation_fn=nn.functional.silu)

        # further fan out channels, with another 2x2-> pool
        self.convblock2 = SimpleConvBlock(16, 64, False, stride=2, activation_fn=nn.functional.silu)
        # keep identifying smaller features, but keep spatial info
        self.res2 = SimpleResidualBlock(64)

        # TBD: strided conv rather than pooling
        self.convblock3 = SimpleConvBlock(64, 64, False, stride=1)
        self.res3 = SimpleResidualBlock(64)
        self.neck3 = DepthwiseSeperableBottleneck(64, 96, 192)

        # Conv layer to narrow channels.
        self.convblock4 = SimpleConvBlock(96, 64, False)
        self.res4 = SimpleResidualBlock(64)

        # Further downsample depth map
        self.convblock5 = SimpleConvBlock(64, 32, False, stride=2)
        self.pre_fc_dropout = nn.Dropout(0.1)

        # Gross. There has to be a smarter way to calculate this dim, right?
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_Y, IMG_X)
            dummy_out = self.fwd_logic(dummy)
            flat_dim = dummy_out.view(1, -1).shape[1]
            self.fc_classifier = nn.Linear(flat_dim, 100)

    def fwd_logic(self, imgdata):
        """Encapsulate forward() logic so it can be referenced in init"""
        # First convolution + rectification +  pool
        imgdata = self.neck1(imgdata)
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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=INITIAL_LR, momentum=INITIAL_MOMENTUM, weight_decay=1e-4
    )
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=INITIAL_LR * 5,           # e.g. 0.05 if INITIAL_LR = 0.01
        steps_per_epoch=len(training_loader),
        epochs=EPOCHS,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4,
        anneal_strategy="cos"
    )



    # The training loop
    for epoch in range(EPOCHS):
        print("Training epoch {} of {}".format(epoch + 1, EPOCHS))
        model.train()
        loss_acc = 0
        for images, labels in training_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            # Note: fixing the depricated warning breaks a test environment
            # due to the change not being implemented on that pytorch version.
            with torch.cuda.amp.autocast(enabled=enable_cuda_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if True:
                scheduler.step()

            loss_acc += loss.item()
        if False:
            scheduler.step() # for cosine annealer that doesn't step every batch

        print(
            "Done training epoch {}/{}, loss for epoch: {}, lr: {}".format(
                epoch + 1, EPOCHS, loss_acc, scheduler.get_last_lr()[0]
            )
        )

        # Print acc on an interval, it's expensive to do every epoch
        if (epoch + 1) % STATS_INTERVAL == 0 and epoch != 0:
            acc = check_acc(model, validation_loader, device)
            print("Epoch {} acc: {}".format(epoch, acc))
        else:
            print("Skipping test check for epoch {}".format(epoch))

    # This isn't tracking the best acc, just the last one
    finalacc = check_acc(model, validation_loader, device)
    print("Overall validation test accuracy: {}".format(finalacc))


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
    p = argparse.ArgumentParser(description="Simple nn training loop")
    p.add_argument("--cpu_num", type=int, default=6, help="CPU cores to use.")
    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=12,
        help="Lower to reduce memory consumption, raise if gpu is starved",
    )
    p.add_argument("--epochs", type=int, default=20, help="Training epochs to execute")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument(
        "--initial_lr", type=float, default=0.01, help="Initial learning rate"
    )
    p.add_argument(
        "--initial_momentum",
        type=float,
        default=0.9,
        help="Not applicable to all optimizers",
    )
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
