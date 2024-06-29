from pathlib import Path
from tempfile import TemporaryDirectory

import rasterio
import numpy as np
import pytest

from ebfloeseg.savefigs import imsave


@pytest.mark.slow
def test_imsave():
    with rasterio.open("tests/input/tci/tci_2012-08-01_214_terra.tiff") as tci:

        img = np.dstack(tci.read()[:])

        with TemporaryDirectory() as temp:
            t = Path(temp)

            # test without res provided
            imsave(tci, img, t, "doy", "fname", count=3)
            assert t.joinpath("fname").exists()

            # test with res provided
            imsave(tci, img, t, "doy", "fnameres", count=3, res="res")
            assert t.joinpath("res_doy_fnameres").exists()

            # test with as_uint8
            imsave(
                tci,
                img[:, :, 1],
                t,
                "doy",
                "fnameuint8",
                count=1,
                as_uint8=True,
                rollaxis=False,
            )
            assert t.joinpath("fnameuint8").exists()
