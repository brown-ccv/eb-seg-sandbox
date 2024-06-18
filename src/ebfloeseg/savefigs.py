from pathlib import Path
from typing import Union

import numpy as np
import rasterio
from rasterio import DatasetReader
from numpy.typing import NDArray


def imsave(
    tci: DatasetReader,
    img: NDArray,
    save_direc: Path,
    doy: str,
    fname: Union[str, Path],
    count: int = 3,
    compress: str = "lzw",
    rollaxis: bool = True,
    as_uint8: bool = False,
    res = None
) -> None:
    with rasterio.Env():
        profile = tci.profile
        profile.update(dtype=rasterio.uint8, count=count, compress="lzw")

        fname = f"{save_direc}{res}_{doy}_{fname}" if res else f"{save_direc}{doy}{fname}"

        with rasterio.open(fname, "w", **profile) as dst:
            if rollaxis:
                img = np.rollaxis(img, axis=2)
                dst.write(img)
                return

            if as_uint8:
                img = img.astype(np.uint8)
                dst.write(img, 1)
                return
