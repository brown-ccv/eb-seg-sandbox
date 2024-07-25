from pathlib import Path

import rasterio
import pandas as pd

from ebfloeseg.utils import getmeta, getres, get_region_properties
from ebfloeseg.masking import create_cloud_mask
from ebfloeseg.savefigs import imsave
from ebfloeseg.preprocess import preprocess


def extract_features(output, red_c, target_dir, res, sat):
    fname = target_dir / f"{res}_{sat}_props.csv"
    props = get_region_properties(output, red_c)
    df = pd.DataFrame.from_dict(props)
    df.to_csv(fname)


def process(
    fcloud,
    ftci,
    fcloud_direc,
    ftci_direc,
    save_figs,
    save_direc,
    land_mask,
    erode_itmax: int = 8,
    erode_itmin: int = 3,
    step: int = -1,
    erosion_kernel_type: str = "diamond",
    erosion_kernel_size: int = 1,
):

    doy, year, sat = getmeta(fcloud)
    res = getres(doy, year)

    target_dir = Path(save_direc, doy)
    target_dir.mkdir(exist_ok=True, parents=True)

    tci = rasterio.open(ftci_direc / ftci)
    cloud_mask = create_cloud_mask(fcloud_direc / fcloud)

    ## OLD
    _cloud=rasterio.open(fcloud)
    _tci=rasterio.open(ftci)
    _cloud_mask=(_cloud.read()[0])==255
    assert cloud_mask.all()==_cloud_mask.all()
    assert tci.read().all()==_tci.read().all()
    assert False
    raise Exception("Boo!")
    asdfasd


    output, red_c = preprocess(
        tci,
        cloud_mask,
        land_mask,
        erosion_kernel_type,
        erosion_kernel_size,
        erode_itmax,
        erode_itmin,
        step,
        save_figs,
        target_dir,
        doy,
        year,
    )

    # saving the props table and label floes tif
    extract_features(output, red_c, target_dir, res, sat)

    # saving the label floes tif
    fname = f"{sat}_final.tif"
    imsave(
        tci,
        output,
        target_dir,
        doy,
        fname,
        count=1,
        rollaxis=False,
        as_uint8=True,
        res=res,
    )
