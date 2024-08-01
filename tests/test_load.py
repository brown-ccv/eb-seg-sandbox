from pathlib import Path

import pytest

from ebfloeseg.load import main, ImageType


def are_equal(p1, p2):
    return Path(p1).read_bytes() == Path(p2).read_bytes()


@pytest.mark.smoke
@pytest.mark.slow
@pytest.mark.parametrize("channel", ImageType)
def test_load(tmpdir, channel):

    filename = f"{channel.value}.tiff"
    main(tmpdir / filename, kind=channel, scale=10000)
    assert are_equal(tmpdir / filename, Path("tests/load/") / filename)
