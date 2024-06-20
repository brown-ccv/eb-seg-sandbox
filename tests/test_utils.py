from ebfloeseg.utils import *
import numpy as np
import tempfile
from pathlib import Path

f1 = "cloud_2012-08-01_214_terra.tiff"
f2 = "tci_2013-08-04_217_terra.tiff"


def test_getdoy():
    assert getdoy(f1) == "214"
    assert getdoy(f2) == "217"


def test_getyear():
    assert getyear(f1) == "2012"
    assert getyear(f2) == "2013"


def test_getsat():
    assert getsat(f1) == "terra"
    assert getsat(f2) == "terra"


def test_getmeta():
    assert getmeta(f1) == ("214", "2012", "terra")
    assert getmeta(f2) == ("217", "2013", "terra")


def test_getres():
    assert getres("214", "2012") == "2012-08-01"
    assert getres("217", "2012") == "2012-08-04"


def test_write_mask_values():
    with tempfile.TemporaryDirectory() as temp_dir:
        land_mask = np.ones((10, 10))
        lmd = np.ones((10, 10))
        ice_mask = np.ones((10, 10))
        doy = "214"
        year = "2012"
        save_direc = Path(temp_dir)
        write_mask_values(
            land_mask.astype(int),
            lmd.astype(int),
            ice_mask.astype(int),
            doy,
            year,
            save_direc,
        )
        assert (save_direc / "mask_values_2012.txt").exists()
        with open(save_direc / "mask_values_2012.txt", "r") as f:
            assert f.readline() == "214\t100\t-200\t-0.5\n"


def test_get_region_properties():
    np.random.seed(42)
    img = np.random.choice([False, True], size=(10, 10))
    red_c = np.ones((10, 10))
    props = get_region_properties(img.astype(int), red_c.astype(int))
    assert "label" in props
    assert "area" in props
    assert "centroid-0" in props


def test_imshow():
    img = np.random.choice([False, True], size=(10, 10))
    imshow(img)
    assert True
