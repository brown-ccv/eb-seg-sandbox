from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import rasterio


def mask_image(img: NDArray, mask: NDArray, val=0) -> NDArray:
    """
    Mask the input image with the given mask inplace.

    Args:
        img (NDArray): The input image to be masked.
        mask (NDArray): The mask to be applied to the image.
        val (int, optional): The value to be assigned to the masked pixels. Defaults to 0.

    Returns:
        NDArray: The masked image.
    """
    img[mask] = val
    return img


def create_land_mask(lmfile: Path, val: int = 75) -> NDArray[np.bool_]:
    """
    Create a land mask from a raster file.

    Parameters:
    lmfile (str): The path to the raster file.

    Returns:
    NDArray[np.bool_]: The land mask as a boolean NumPy array.
    """
    s = rasterio.open(lmfile)
    land_mask = (s.read()[0]) == val
    return land_mask


def create_cloud_mask(cloud_file: Path, val: int = 255) -> NDArray[np.bool_]:
    """
    Create cloud mask from cloud file.

    Args:
        cloud_file (Path): path to cloud (raster) file

    Returns:
        NDArray[np.bool_]: cloud mask
    """
    cloud_mask = create_land_mask(cloud_file, val)
    return cloud_mask


def maskrgb(rgb: NDArray, mask: NDArray) -> None:
    """
    Apply (inplace) a mask to each channel of an RGB image.

    Args:
        rgb (numpy.ndarray): The RGB image to be masked.
        mask (numpy.ndarray): The mask to be applied.
    """
    for i in range(rgb.shape[2]):
        mask_image(rgb[:, :, i], mask)
