import torch
from torch.nn import Sequential, Conv2d, LeakyReLU, Sigmoid, Flatten, BatchNorm2d
from torch.utils.data import Dataset

from Entities.Extensions import device


class Discriminator(torch.Module):
    Dis: Sequential
    model_name: str

    training_set: Dataset
    test_set: Dataset
    work_set: Dataset

    def __init__(self):
        self.init_model()

    @staticmethod
    def static_init_model() -> Sequential:
        Dis = torch.nn.Sequential(
            Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(64),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(128),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, bias=False),
            Sigmoid(),
            Flatten()
        ).to(device=device)
        return Dis

    @classmethod
    def init_model(cls) -> None:
        cls.Dis = torch.nn.Sequential(
            Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(64),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(128),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, bias=False),
            Sigmoid(),
            Flatten()
        ).to(device=device)

    @classmethod
    def train_model(cls):
        pass

    @classmethod
    def work_model(cls):
        pass

    @classmethod
    def set_name(cls, name: str):
        cls.model_name = name
