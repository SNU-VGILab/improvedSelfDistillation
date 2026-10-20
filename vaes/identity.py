import torch
from torchvision import transforms


class IDAE:
    def __init__(self, name: str | None = None, img_size: int = 256):
        self.img_size = img_size

    def img_transform(self, p_hflip: float = 0, img_size: int | None = None):
        img_transforms = [
            transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
        return transforms.Compose(img_transforms)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return images

    def decode_to_images(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.clip((z + 1) / 2, 0, 1)
