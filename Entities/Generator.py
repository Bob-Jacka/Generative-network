import matplotlib.pyplot as plt
import torch.cuda
import torchvision
from torch import T
from torch.ao.nn.quantized import ConvTranspose2d
from torch.nn import Sequential, BatchNorm2d, ReLU, Tanh
from torch.utils.data import Dataset

from Entities.Extensions import device, transform_func


class Generator(torch.Module):
    Gen: Sequential
    model_name: str

    training_set: Dataset
    test_set: Dataset
    work_set: Dataset

    def __init__(self):
        self.init_model()

    @staticmethod
    def static_init_model() -> Sequential:
        Gen = torch.nn.Sequential(
            ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(512),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(256),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(128),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, bias=False),
            Tanh()
        ).to(device=device)
        return Gen

    @classmethod
    def init_model(cls) -> None:
        cls.Gen = torch.nn.Sequential(
            ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(512),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(256),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(128),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, bias=False),
            Tanh()
        ).to(device=device)

    @classmethod
    def train_model(cls):
        cls.training_set = torchvision.datasets.MNIST(root='.', train=True, download=True, transform=transform_func)

    @classmethod
    def test_model(cls):
        cls.test_set = torchvision.datasets.MNIST(root='.', train=False, download=True, transform=transform_func)

    @classmethod
    def work_model(cls):
        pass

    @classmethod
    def set_name(cls, name: str):
        cls.model_name = name

    @classmethod
    def see_output(cls):
        noise = torch.randn(32, 100).to(device=device)
        fake_samples = cls.Gen(noise).cpu().detach()
        plt.figure(dpi=100, figsize=(20, 10))
        for i in range(32):
            ax = plt.subplot(4, 8, i + 1)
            img = (fake_samples[i] / 2 + 0.5).reshape(28, 28)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def train_G(self):
        pass
