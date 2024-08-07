from pathlib import Path

import pytest

from ebfloeseg.app import load, ImageType, Satellite


def are_equal(p1, p2):
    return Path(p1).read_bytes() == Path(p2).read_bytes()


@pytest.mark.smoke
@pytest.mark.slow
@pytest.mark.parametrize("kind", ImageType)
def test_load(tmpdir, kind):

    filename = f"{kind.value}.tiff"
    load(tmpdir / filename, kind=kind, scale=10000)
    assert are_equal(tmpdir / filename, Path("tests/load/") / filename)


@pytest.mark.slow
@pytest.mark.parametrize("satellite", Satellite)
@pytest.mark.parametrize("kind", ImageType)
def test_load(tmp_path, kind, satellite):
    load(tmp_path / "out.tiff", kind=kind, satellite=satellite, scale=100000)
