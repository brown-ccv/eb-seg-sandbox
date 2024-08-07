from pathlib import Path
import subprocess
from collections import defaultdict

import pytest
import pandas as pd

from ebfloeseg.app import parse_config_file


def are_equal(p1, p2):
    return Path(p1).read_bytes() == Path(p2).read_bytes()


def check_sums(p1, p2):
    s1 = pd.read_csv(p1).to_numpy().sum()
    s2 = pd.read_csv(p2).to_numpy().sum()
    return s1 == s2


# Check final images
def _test_output(tmpdir):
    expdir = Path("tests/expected")
    # Check final images
    # -----------------------------------------------------------------
    f214 = tmpdir / "214" / "2012-08-01_terra_final.tif"
    f214expected = expdir / "214/2012-08-01_214_terra_final.tif"
    assert are_equal(f214, f214expected)
    f215expected = expdir / "215/2012-08-02_215_terra_final.tif"
    f215 = tmpdir / "215/2012-08-02_terra_final.tif"
    assert are_equal(f215, f215expected)

    # Check mask values
    # -----------------------------------------------------------------
    maskvalues214 = tmpdir / "214/mask_values.txt"
    maskvalues214expected = expdir / "214/mask_values.txt"
    assert are_equal(maskvalues214, maskvalues214expected)

    maskvalues215 = tmpdir / "215/mask_values.txt"
    maskvalues215expected = expdir / "215/mask_values.txt"
    assert are_equal(maskvalues215, maskvalues215expected)

    # Check feature extraction
    # -----------------------------------------------------------------
    features214 = tmpdir / "214/2012-08-01_terra_props.csv"
    features214expected = expdir / "214/2012-08-01_terra_props.csv"
    assert check_sums(features214, features214expected)

    features215 = tmpdir / "215/2012-08-02_terra_props.csv"
    features215expected = expdir / "215/2012-08-02_terra_props.csv"
    assert check_sums(features215, features215expected)

    # Check intermediate identification rounds
    # -----------------------------------------------------------------
    for doy in ["214", "215"]:
        id_rounds = sorted(Path(tmpdir / doy).glob("*round*.tif"))
        expected_rounds = sorted((expdir / doy).glob("*round*.tif"))

        for id_round, expected_round in zip(id_rounds, expected_rounds):
            assert are_equal(id_round, expected_round)


def getmaskvalues(path):
    with open(path, "r") as f:
        lines = f.readlines()

    mask_values = [float(x) for x in lines[0].split()]
    return mask_values


def group_files_by_extension(folder_path):
    grouped_files = defaultdict(list)
    folder = Path(folder_path)

    for file in folder.iterdir():
        if file.is_file():
            extension = file.suffix
            grouped_files[extension].append(file.name)

    return dict(grouped_files)


@pytest.mark.smoke
@pytest.mark.slow
def test_fsdproc(tmpdir):
    expdir = Path("tests/expected")
    config_file = tmpdir.join("config.toml")
    config_file.write(
        f"""
        data_direc = "tests/input"
        save_figs = true
        save_direc = "{tmpdir}"
        land = "tests/input/reproj_land.tiff"
        [erosion]
        itmax = 8
        itmin = 3
        step = -1
        kernel_type = "diamond"
        kernel_size = 1
        """
    )

    result = subprocess.run(
        [
            "fsdproc",
            "process-batch",
            "--config-file",
            str(config_file),
            "--max-workers",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    # Check command ran successfully
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"

    _test_output(tmpdir)


def test_parse_config_file(tmpdir):
    config_file = tmpdir.join("config.toml")
    config_file.write(
        """
        data_direc = "/path/to/data"
        save_figs = true
        save_direc = "/path/to/save"
        land = "/path/to/landfile"
        [erosion]
        itmax = 10
        itmin = 5
        step = 2
        kernel_type = "ellipse"
        kernel_size = 3
        """
    )

    params = parse_config_file(config_file)

    assert params.data_direc == Path("/path/to/data")
    assert params.save_figs
    assert params.save_direc == Path("/path/to/save")
    assert params.land == Path("/path/to/landfile")
    assert params.itmax == 10
    assert params.itmin == 5
    assert params.step == 2
    assert params.kernel_type == "ellipse"
    assert params.kernel_size == 3
