import torch
import numpy as np
from PIL import Image


def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)

def scale_uv(
    uv             : torch.Tensor,
    border_padding : float = 0.0,
    deform         : bool = False,
):
    min_uv, _ = torch.min(uv, dim=-2, keepdim=True)
    max_uv, _ = torch.max(uv, dim=-2, keepdim=True)
    max_extend = max_uv - min_uv
    if not deform:
        max_extend, _ = torch.max(max_uv - min_uv, dim=-1, keepdim=True)

    scaling = (1.0 - 2.0 * border_padding) / max_extend
    return (uv - min_uv) * scaling + border_padding + 0.5 * (max_extend - (max_uv - min_uv)) * scaling


def loadTexture(
    texture_file : str,
    texture_size : torch.Tensor,
) -> torch.Tensor:
    texture = Image.open(texture_file)
    if torch.all(texture_size >= 0):
        texture = texture.resize(texture_size, Image.Resampling.BICUBIC)
    texture = torch.from_numpy(np.array(texture, dtype = np.float32)).to("cuda:0").unsqueeze(0) / 255.0
    texture = torch.flip(texture, dims = [1])
    texture = torch.cat((texture, torch.ones((*(texture.shape[:-1]), 1), device="cuda:0")), dim=-1)
    return texture

def floatTensorToImage(
    float_image : torch.Tensor,
) -> torch.Tensor:
    uint_image = torch.clamp(torch.round(float_image.detach() * 255), 0, 255)
    uint_image = uint_image.type(torch.uint8)
    return uint_image