from pathlib import Path
from tempfile import TemporaryDirectory

import rasterio
import numpy as np

from ebfloeseg.savefigs import imsave


def test_imsave():
    tci = rasterio.open("tests/input/tci/tci_2012-08-01_214_terra.tiff")

    img = np.dstack(tci.read()[:])

    temp = TemporaryDirectory()

    # test without res provided
    imsave(tci, img, Path(temp.name), "doy", "fname", count=3)
    assert Path(temp.name).joinpath("fname").exists()

    # test with res provided
    imsave(tci, img, Path(temp.name), "doy", "fnameres", count=3, res="res")
    assert Path(temp.name).joinpath("res_doy_fnameres").exists()

    # test with as_uint8
    imsave(
        tci,
        img[:, :, 1],
        Path(temp.name),
        "doy",
        "fnameuint8",
        count=1,
        as_uint8=True,
        rollaxis=False,
    )
    assert Path(temp.name).joinpath("fnameuint8").exists()

    temp.cleanup()
