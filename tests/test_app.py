from collections import Counter
import os
from pathlib import Path
import subprocess
from collections import defaultdict

import pytest
import pandas as pd
import numpy as np

from ebfloeseg.app import parse_config_file
from ebfloeseg.utils import imopen

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
            "--max-workers",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    # Check command ran successfully
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"


    # Check output files were created
    for folder in ["214", "215"]:
        gen = Path(tmpdir / folder)
        exp = Path(expdir / folder)
        genfiles = group_files_by_extension(gen)
        expfiles = group_files_by_extension(exp)

        for ext in genfiles:
            for file in genfiles[ext]:
                if ext == ".txt":
                    genmask = getmaskvalues(gen / file)
                    expmask = getmaskvalues(exp / file)
                    assert genmask == expmask, f"Mask values in {file} do not match"

                else:
                    imgen = imopen(gen / file)
                    imexp = imopen(exp / file)
                    assert np.array_equal(imexp, imgen)

    # expected_counts_by_extension = {".tif": 10, ".txt": 1, ".csv": 1, ".png": 1}

    # for ext, count in Counter(
    #     Path(f).suffix for f in os.listdir(tmpdir / "214")
    # ).items():
    #     assert expected_counts_by_extension[ext] == count

    # # check images pixel-wise.
    # # 214 images and files
    # assert (tmpdir / "214" / "214_1.tif").read_bytes() == (
    #     Path("tests/expected_output/214/214_1.tif").read_bytes()
    # )


    # assert False

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
