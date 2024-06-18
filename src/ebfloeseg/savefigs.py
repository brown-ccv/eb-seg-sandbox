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
) -> None:
    with rasterio.Env():
        profile = tci.profile
        profile.update(
            dtype=rasterio.uint8,
            count=count,
            compress=compress,
        )

        if rollaxis:
            img = np.rollaxis(img, axis=2)

        if as_uint8:
            img = img.astype(rasterio.uint8)

        _filename = Path(save_direc) / fname
        with rasterio.open(_filename, "w", **profile) as dst:
            dst.write(img)