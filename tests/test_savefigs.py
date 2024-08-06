import rasterio
import numpy as np
import pytest

from ebfloeseg.savefigs import imsave


@pytest.mark.slow
def test_imsave(tmp_path):
    with rasterio.open("tests/input/tci/tci_2012-08-01_214_terra.tiff") as tci:

        img = np.dstack(tci.read()[:])

        # test without res provided
        imsave(tci, img, tmp_path, "fname", count=3)
        assert tmp_path.joinpath("fname").exists()

        # test with res provided
        imsave(tci, img, tmp_path, "fnameres", count=3, res="res")
        assert tmp_path.joinpath("res_fnameres").exists()

        # test with as_uint8
        imsave(
            tci,
            img[:, :, 1],
            tmp_path,
            "fnameuint8",
            count=1,
            as_uint8=True,
            rollaxis=False,
        )
        assert tmp_path.joinpath("fnameuint8").exists()
