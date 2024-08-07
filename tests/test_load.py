from pathlib import Path
from io import BytesIO

import pytest
import requests_mock

from ebfloeseg.load import load, ImageType, Satellite


def are_equal(b1: BytesIO, p2):
    return b1.read() == Path(p2).read_bytes()


@pytest.mark.smoke
@pytest.mark.slow
@pytest.mark.parametrize("kind", ImageType)
def test_load(kind):
    result = load(kind=kind, scale=10000)
    data = BytesIO(result["content"])
    assert are_equal(data, Path("tests/load/") / f"{kind.value}.tiff")


@pytest.mark.slow
@pytest.mark.parametrize("satellite", Satellite)
@pytest.mark.parametrize("kind", ImageType)
def test_load(kind, satellite):
    load(kind=kind, satellite=satellite, scale=100000)


def test_error_on_empty_file():
    with requests_mock.Mocker() as m:
        m.get(
            "https://wvs.earthdata.nasa.gov/api/v1/snapshot",
            content=Path("tests/load/empty.tiff").read_bytes(),
        )
        with pytest.raises(AssertionError):
            load()
