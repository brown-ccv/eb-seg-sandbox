from datetime import datetime

import numpy as np
from numpy.typing import NDArray



def mask_image(img: NDArray, mask: NDArray, val=0) -> NDArray:
    img[mask] = val


# test mask_image
img = np.ones((5, 5))
mask = np.array(
    (
        [
            [True, False, True, False, False],
            [False, False, True, True, False],
            [False, True, False, True, True],
            [False, False, True, True, False],
            [False, True, True, True, False],
        ]
    )
)
mask_image(img, mask)
arr = np.array(
    (
        [
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
)
assert np.allclose(img, arr)


def getdoy(fname: str) -> str:
    return fname.split("_")[-2].zfill(3)


f1 = "cloud_2012-08-01_214_terra.tiff"
f2 = "tci_2012-08-04_217_terra.tiff"
assert getdoy(f1) == "214"
assert getdoy(f2) == "217"


def getyear(fname: str) -> str:
    return fname.split("_")[1].split("-")[0]


def getsat(fname: str) -> str:
    return fname.split("_")[-1].split(".")[0]


assert getyear(f1) == "2012"
assert getyear(f2) == "2012"

assert getsat(f1) == "terra"
assert getsat(f2) == "terra"


def getmeta(fname: str) -> str:
    doy = getdoy(fname)
    year = getyear(fname)
    sat = getsat(fname)
    return doy, year, sat


assert getmeta(f1) == ("214", "2012", "terra")
assert getmeta(f2) == ("217", "2012", "terra")


def getres(doy: str, year: str) -> str:
    return datetime.strptime(year + "-" + doy, "%Y-%j").strftime("%Y-%m-%d")

assert getres("214", "2012") == "2012-08-01"
assert getres("217", "2012") == "2012-08-04"