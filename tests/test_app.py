import subprocess
from pathlib import Path
import pytest
from ebfloeseg.app import parse_config_file
from collections import Counter
import os


@pytest.mark.smoke
@pytest.mark.slow
def test_fsdproc(tmpdir):
    config_file = tmpdir.join("config.toml")
    config_file.write(
        f"""
        data_direc = "tests/input"
        save_figs = true
        save_direc = "{tmpdir}"
        land = "tests/input/reproj_land.tiff"
        [erosion]
        erode_itmax = 8
        erode_itmin = 3
        step = -1
        erosion_kernel_type = "diamond"
        erosion_kernel_size = 1
        """
    )

    result = subprocess.run(
        [
            "fsdproc",
            "--config-file",
            str(config_file),
        ],
        capture_output=True,
        text=True,
    )

    # Check command ran successfully
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"

    # Check output files were created
    for folder in ["214", "215"]:
        assert any(
            Path(tmpdir, folder).iterdir()
        ), f"No files were created in the output directory {folder}."

    expected_counts_by_extension = {".tif": 10, ".txt": 1, ".csv": 1, ".png": 1}

    for ext, count in Counter(
        Path(f).suffix for f in os.listdir(tmpdir / "214")
    ).items():
        assert expected_counts_by_extension[ext] == count


def test_parse_config_file(tmpdir):
    config_file = tmpdir.join("config.toml")
    config_file.write(
        """
        data_direc = "/path/to/data"
        save_figs = true
        save_direc = "/path/to/save"
        land = "/path/to/landfile"
        [erosion]
        erode_itmax = 10
        erode_itmin = 5
        step = 2
        erosion_kernel_type = "ellipse"
        erosion_kernel_size = 3
        """
    )

    params = parse_config_file(config_file)

    assert params.data_direc == Path("/path/to/data")
    assert params.save_figs
    assert params.save_direc == Path("/path/to/save")
    assert params.land == Path("/path/to/landfile")
    assert params.erode_itmax == 10
    assert params.erode_itmin == 5
    assert params.step == 2
    assert params.erosion_kernel_type == "ellipse"
    assert params.erosion_kernel_size == 3
