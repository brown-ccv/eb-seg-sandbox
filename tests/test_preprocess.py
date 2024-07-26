import pytest
from pathlib import Path
from ebfloeseg.preprocess import preprocess


def test_process_exception(tmpdir):
    fcloud = "cloud.tif"
    ftci = "tci.tif"
    fcloud_direc = Path(tmpdir, "clouds")
    ftci_direc = Path(tmpdir, "tcis")
    save_figs = True
    save_direc = tmpdir
    land_mask = "land_mask.tif"
    erode_itmax = 8
    erode_itmin = 3
    step = -1
    erosion_kernel_type = "diamond"
    erosion_kernel_size = 1

    with pytest.raises(Exception):
        preprocess(
            fcloud,
            ftci,
            fcloud_direc,
            ftci_direc,
            save_figs,
            save_direc,
            land_mask,
            erode_itmax,
            erode_itmin,
            step,
            erosion_kernel_type,
            erosion_kernel_size,
        )
