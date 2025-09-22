import torch
from dataclasses import dataclass

@dataclass
class dataConfig:
    data_path: str = "./data/brain_tumor"
    model_name: str = "google/vit-base-patch16-224-in21k"
    batch_size: int = 32
    epochs: int = 5
    lr: float = 2e-5
    num_classes: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus: int = torch.cuda.device_count()
    device_name :str = torch.cuda.get_device_name()