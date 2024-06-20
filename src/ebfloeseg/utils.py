from datetime import datetime

import matplotlib.pyplot as plt
import skimage


def imshow(img, cmap="gray"):
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()


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


def write_mask_values(land_mask, lmd, ice_mask, doy, year, save_direc):
    land_cloud_mask_sum = sum(sum(~(lmd)))
    fname = save_direc / f"mask_values_{year}.txt"
    ice_mask_sum = sum(sum(ice_mask))
    ratio = ice_mask_sum / land_cloud_mask_sum
    towrite = f"{doy}\t{ice_mask_sum}\t{land_cloud_mask_sum}\t{ratio}\n"
    with open(fname, "a") as f:
        f.write(towrite)


def get_region_properties(img, red_c):
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
