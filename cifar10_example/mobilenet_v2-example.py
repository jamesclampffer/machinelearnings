import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader


def main():
    torch.set_num_threads(8)

    # Runtime config
    BATCH_SIZE = 128
    EPOCHS = 10

    # Transforms
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),  # Fuzz colors and brightness
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Datasets
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=100)
    model.to(device)

    # Optimizer, Loss, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(
            f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    total += targets.size(0)

            acc = 100.0 * correct / total
            print(f"Validation Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
