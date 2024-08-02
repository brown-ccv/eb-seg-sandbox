import numpy as np

from ebfloeseg.utils import (
    write_mask_values,
    get_region_properties,
    imshow,
    getdoy,
    getyear,
    getsat,
    getmeta,
    getres,
)

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


def test_write_mask_values(tmpdir):
    lmd = np.ones((3, 3))
    ice_mask = np.ones((3, 3))
    doy = "214"
    write_mask_values(
        lmd=lmd.astype(int),
        ice_mask=ice_mask.astype(int),
        doy=doy,
        save_direc=tmpdir,
        fname="mask_values.txt",
    )

    with open(tmpdir / "mask_values.txt", "r") as f:
        assert f.readline() == "214\t9\t-18\t-0.5\n"


def test_get_region_properties():
    img = np.ones((3, 3))
    red_c = np.ones((3, 3))
    props = get_region_properties(img, red_c)
    assert "label" in props
    assert "area" in props
    assert "centroid-0" in props


def test_imshow():
    img = np.random.choice([False, True], size=(1, 1))
    imshow(img, show=False)
    assert True
