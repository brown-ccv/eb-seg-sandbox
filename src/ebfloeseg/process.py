from pathlib import Path

import rasterio
import pandas as pd

from ebfloeseg.utils import getmeta, getres, get_region_properties
from ebfloeseg.masking import create_cloud_mask
from ebfloeseg.savefigs import imsave
from ebfloeseg.preprocess import preprocess


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
    props = get_region_properties(output, red_c)

    df = pd.DataFrame.from_dict(props)
    fname = target_dir / f"{res}_{sat}_props.csv"
    df.to_csv(fname)

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
