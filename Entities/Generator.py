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
        transform = T.compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
        cls.test_set = torchvision.datasets.MNIST(root='.', train=False, download=True, transform=transform_func)

    @classmethod
    def work_model(cls):
        pass

    @classmethod
    def set_name(cls, name: str):
        cls.model_name = name

    @classmethod
    def test_epoch(cls):
        how_many = 32
        noise = torch.randn(how_many, 100, 1, 1).to(device)
        fsamples = cls.Gen(noise).cpu().detach()
        for sample in range(how_many):
            ax = plt.subplots(4, 8, sample + 1)
            img = (fsamples.cpu().detach()[sample] / 2 + 0.5).permulate(1, 2, 0)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
        plt.subplots_adjust(hspace=-0.6)
        plt.show()
