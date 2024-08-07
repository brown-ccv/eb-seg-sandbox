from pathlib import Path
from io import BytesIO

import pytest
import requests_mock

from ebfloeseg.io_ import load, ImageType


def are_equal(b1: BytesIO, p2):
    return b1.read() == Path(p2).read_bytes()


@pytest.mark.smoke
@pytest.mark.slow
@pytest.mark.parametrize("channel", ImageType)
def test_load(channel):
    result = load(kind=channel, scale=10000)
    data = BytesIO(result["content"])
    assert are_equal(data, Path("tests/load/") / f"{channel.value}.tiff")


def test_error_on_empty_file():
    with requests_mock.Mocker() as m:
        m.get(
            "https://wvs.earthdata.nasa.gov/api/v1/snapshot",
            content=Path("tests/load/empty.tiff").read_bytes(),
        )
        with pytest.raises(AssertionError):
            load()
