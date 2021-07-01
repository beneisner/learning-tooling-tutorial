# Adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Adapted from https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import typer


class LightningNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        # NOTICE: DON'T NEED TO OPTIMIZE, CLEAR, ETC.
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        if self.training_step % 2000 == 1999:
            print("[%d] loss: %.3f" % (self.training_epoch, loss))

        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer


def main(batch_size: int):
    # Define the data.
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Define the data.
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # Just create the net!
    net = LightningNet()

    # This is the entire training loop!
    trainer = pl.Trainer(max_epochs=100)
    # trainer = pl.Trainer(gpus=1)  # ALL YOU NEED TO DO TO ADD A DEVICE!

    # This runs the whole training
    trainer.fit(net, trainloader)


if __name__ == "__main__":
    typer.run(main)
