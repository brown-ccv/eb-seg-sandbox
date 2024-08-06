from pathlib import Path
from typing import Union

import numpy as np
import rasterio
from rasterio import DatasetReader
from numpy.typing import NDArray
from matplotlib import pyplot as plt


def imsave(
    tci: DatasetReader,
    img: NDArray,
    save_direc: Path,
    fname: Union[str, Path],
    count: int = 3,
    compress: str = "lzw",
    rollaxis: bool = True,
    as_uint8: bool = False,
    res=None,
) -> None:
    profile = tci.profile
    profile.update(
        dtype=rasterio.uint8,  # sample images are uint8; might not be needed? CP
        count=count,
        compress=compress,
    )

    if res:
        fname = save_direc / f"{res}_{fname}"
    else:
        fname = save_direc / fname

    with rasterio.open(fname, "w", **profile) as dst:
        if rollaxis:
            img = np.rollaxis(img, axis=2)
            dst.write(img)
            return

        if as_uint8:
            img = img.astype(np.uint8)
            dst.write(img, 1)


def save_ice_mask_hist(
    red_masked,
    bins,
    mincut,
    maxcut,
    target_dir,
    fname: Union[str, Path],
    color="r",
    figsize=(6, 2),
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.hist(red_masked.flatten(), bins=bins, color=color)
    plt.axvline(mincut)
    plt.axvline(maxcut)
    plt.savefig(target_dir / fname)
    return ax
