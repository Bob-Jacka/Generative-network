import torch
from torch import T
from torch.utils.data import DataLoader

save_path: str = "../save_model/"
device: str = "cuda" if torch.cuda.is_available() else "cpu"
transform_func = T.compose([T.ToTensor(), T.Normalize([0.5], [0.5])])


def save_model(model_to_save):
    scripted = torch.jit.script(model_to_save)
    scripted.save("save_model")


def load_model(model_to_load):
    scripted = torch.jit.load(save_path + model_to_load, map_location=device)
    scripted.eval()


def get_dataloader(dataset, batch_size=1, shuffle: bool = False) -> DataLoader:
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
