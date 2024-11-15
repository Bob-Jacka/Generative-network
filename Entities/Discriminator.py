import torch
import torchvision.datasets
from torch.nn import Sequential, Conv2d, LeakyReLU, Sigmoid, Flatten, BatchNorm2d, BCELoss
from torch.optim import Adam
from torch.utils.data import Dataset

from Entities.Extensions import device, transform_func, get_dataloader
from Entities.Generator import Generator


class Discriminator(torch.Module):
    Dis: Sequential
    optim: Adam
    loss_fn: BCELoss
    model_name: str

    training_set: Dataset
    test_set: Dataset
    work_set: Dataset
    batch_size: int = 32

    real_labels = torch.ones((batch_size, 1)).to(device)
    fake_labels = torch.zeros((batch_size, 1)).to(device)

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
        cls.training_set = torchvision.datasets.FashionMNIST(root=".", train=True, download=True,
                                                             transform=transform_func)
        train_loader = get_dataloader(cls.training_set, cls.batch_size, shuffle=True)

    @classmethod
    def work_model(cls):
        pass

    @classmethod
    def set_name(cls, name: str):
        cls.model_name = name

    @classmethod
    def train_D_on_real(cls, real_samples):
        r = real_samples.reshape(-1, 28 * 28).to(device)
        out_D = cls.Dis(r)
        labels = torch.ones((r.shape[0], 1)).to(device)
        loss_D = cls.loss_fn(out_D, labels)
        cls.optim.zero_grad()
        loss_D.backward()
        cls.optim.step()
        return loss_D

    @classmethod
    def train_D_on_fake(cls, gen: Generator):
        noise = torch.randn(cls.batch_size, 100).to(device=device)
        generated_data = gen(noise)
        preds = cls.Dis(generated_data)
        loss_D = cls.loss_fn(preds, cls.fake_labels)
        cls.optim.zero_grad()
        loss_D.backward()
        cls.optim.step()
        return loss_D
