from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import rasterio


def mask_image(img: NDArray, mask: NDArray, val=0) -> NDArray:
    img[mask] = val


def create_land_mask(lmfile: str) -> NDArray[np.bool_]:
    s = rasterio.open(lmfile)
    return (s.read()[0]) == 75


def create_cloud_mask(cloud_file: Path) -> NDArray[np.bool_]:
    """
    Create cloud mask from cloud file.

    Args:
        cloud_file (Path): path to cloud file

    Returns:
        NDArray[np.bool_]: cloud mask
    """
    cloud = rasterio.open(cloud_file)
    cloud_mask = (cloud.read()[0]) == 255
    return cloud_mask


def maskrgb(rgb, mask):
    for i in range(rgb.shape[2]):
        mask_image(rgb[:, :, i], mask)
