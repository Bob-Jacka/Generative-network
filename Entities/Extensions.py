import torch
from torch import T
from torch.nn import Sequential
from torch.utils.data import DataLoader

save_path: str = "../save_model/"
device: str = "cuda" if torch.cuda.is_available() else "cpu"
transform_func = T.compose([T.ToTensor(), T.Normalize([0.5], [0.5])])


def save_model(model_to_save) -> None:
    scripted = torch.jit.save(model_to_save)
    #TODO дописать


def load_model(model_to_load) -> Sequential:
    scripted = torch.jit.load(save_path + model_to_load, map_location=device)
    loaded_model = scripted[0]
    return loaded_model


def get_dataloader(dataset, batch_size=1, shuffle: bool = False) -> DataLoader:
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
