from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage
from numpy.typing import ArrayLike

from ebfloeseg.peakdet import peakdet


def imshow(img: ArrayLike, cmap: str = "gray", show: bool = True) -> None:
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    if show:
        plt.show()


def imopen(path: str) -> None:
    return plt.imread(path)


def getdoy(fname: str) -> str:
    """
    Extracts the day of year (DOY) from a given filename.

    Args:
        fname (str): The filename from which to extract the DOY.

    Returns:
        str: The day of year (DOY) as a zero-padded string.

    Example:
        >>> getdoy("image_2022_123.jpg")
        '123'
    """
    return fname.split("_")[-2].zfill(3)


def getyear(fname: str) -> str:
    """
    Extracts the year from a given filename.

    Args:
        fname (str): The filename from which to extract the year.

    Returns:
        str: The extracted year.

    Example:
        >>> getyear("file_2022-01-01.txt")
        '2022'
    """
    return fname.split("_")[1].split("-")[0]


def getsat(fname: str) -> str:
    """
    Extracts the satellite name from a given filename.

    Args:
        fname (str): The filename from which to extract the satellite name.

    Returns:
        str: The extracted satellite name.

    """
    return fname.split("_")[-1].split(".")[0]


def getmeta(fname: str | Path) -> tuple[str, str, str]:
    """
    Retrieves the metadata information from the given file name.

    Parameters:
    fname (str): The file name from which to retrieve the metadata.

    Returns:
    tuple[str, str, str]: A tuple containing the day of year (doy), year, and satellite information.
    """
    if isinstance(fname, Path):
        fname = str(fname)

    doy = getdoy(fname)
    year = getyear(fname)
    sat = getsat(fname)
    return doy, year, sat


def getres(doy: str, year: str) -> str:
    return datetime.strptime(year + "-" + doy, "%Y-%j").strftime("%Y-%m-%d")


def write_mask_values(
    lmd: ArrayLike,
    ice_mask: ArrayLike,
    doy: str,
    save_direc: str,
    fname: str,
) -> None:
    """
    Write mask values to a text file.

    Args:
        lmd (numpy.ndarray): Land cloud mask array.
        ice_mask (numpy.ndarray): Ice mask array.
        doy (int): Day of year.
        save_direc (str): Directory to save the text file.

    Returns:
        None
    """
    land_cloud_mask_sum = sum(sum(~(lmd)))
    ice_mask_sum = sum(sum(ice_mask))
    ratio = ice_mask_sum / land_cloud_mask_sum
    towrite = f"{doy}\t{ice_mask_sum}\t{land_cloud_mask_sum}\t{ratio}\n"
    with open(save_direc / fname, "a") as f:
        f.write(towrite)


def get_region_properties(img: ArrayLike, red_c: ArrayLike) -> dict[str, ArrayLike]:
    """
    Calculate properties of regions in an image.

    Parameters:
    - img: ArrayLike
        The input image.
    - red_c: ArrayLike
        The red channel value used for regionprops calculation.

    Returns:
    - props: dict
        A dictionary containing the calculated properties for each region.
    """
    props = skimage.measure.regionprops_table(
        img.astype(int),
        red_c,
        properties=[
            "label",
            "area",
            "centroid",
            "axis_major_length",
            "axis_minor_length",
            "orientation",
            "perimeter",
            "intensity_mean",
        ],
    )
    return props


def get_wcuts(red_masked):
    bins = np.arange(1, 256, 5)
    rn, rbins = np.histogram(red_masked.flatten(), bins=bins)
    dx = 0.01 * np.mean(rn)
    rmaxtab, rmintab = peakdet(rn, dx)
    rmax_n = rbins[rmaxtab[-1, 0]]
    rhm_high = rmaxtab[-1, 1] / 2

    if ~np.any(rmintab):
        ow_cut_min = 100
    else:
        ow_cut_min = rbins[rmintab[-1, 0]]

    if np.any(np.where((rbins[:-1] < rmax_n) & (rn <= rhm_high))):
        ow_cut_max = rbins[
            np.where((rbins[:-1] < rmax_n) & (rn <= rhm_high))[0][-1]
        ]  # fwhm to left of ice max
    else:
        ow_cut_max = rmax_n - 10

    return ow_cut_min, ow_cut_max, bins
