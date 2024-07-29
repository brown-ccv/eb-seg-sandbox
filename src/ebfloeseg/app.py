#!/usr/bin/env python

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import tomllib
import typer
from typing import Annotated, Optional


import pandas as pd

from ebfloeseg.masking import create_land_mask
from ebfloeseg.preprocess import preprocess


@dataclass
class ConfigParams:
    data_direc: Path
    land: Path
    save_figs: bool
    save_direc: Path
    itmax: int
    itmin: int
    step: int
    kernel_type: str
    kernel_size: int


def validate_kernel_type(ctx: typer.Context, value: str) -> str:
    if value not in ["diamond", "ellipse"]:
        raise typer.BadParameter("Kernel type must be 'diamond' or 'ellipse'")
    return value


help = "TODO: add description"
name = "fsdproc"
epilog = f"Example: {name} --data-direc input/ input/land.tiff output/ --save-figs --itmax 8 --itmin 3 --kernel-type diamond"
app = typer.Typer(name=name, add_completion=False)


def parse_config_file(config_file: Path) -> ConfigParams:

    if not config_file.exists():
        raise FileNotFoundError("Configuration file does not exist.")

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    defaults = {
        "data_direc": None,  # directory containing TCI and cloud images
        "save_direc": None,  # directory to save figures
        "land": None,  # path to land mask image
        "save_figs": False,  # whether to save figures
        "itmax": 8,  # maximum number of iterations for erosion
        "itmin": 3,  # (inclusive) minimum number of iterations for erosion
        "step": -1,
        "kernel_type": "diamond",  # type of kernel (either diamond or ellipse)
        "kernel_size": 1,
    }

    erosion = config["erosion"]
    config.pop("erosion")
    config.update(erosion)

    for key in defaults:
        if key in config:
            value = config[key]
            if "direc" in key or key == "land":  # Handle paths specifically
                value = Path(value)
            defaults[key] = value

    return ConfigParams(**defaults)

class KernelType(str, Enum):
    diamond = "diamond"
    ellipse = "ellipse"


@app.command(name="process-images", help=help, epilog=epilog)
def process_images(
    
    in_dir: Annotated[Path, typer.Argument(help="directory containing input files")],
    mask: Annotated[Path, typer.Argument(help="path to land mask file")],
    out_dir: Annotated[Path, typer.Argument(help="directory for outputs")],
    save_figs: Annotated[bool, typer.Option(..., "--save-figs", help="save figures")] = False,
    itmax: Annotated[int, typer.Option(..., "--itmax", help="maximum number of iterations for erosion")] = 8,
    itmin: Annotated[int, typer.Option(..., "--itmin", help="minimum number of iterations for erosion")] = 3,
    step: Annotated[int, typer.Option(..., "--step")] = -1,
    kernel_type: Annotated[KernelType, typer.Option(..., "--kernel-type")] = KernelType.diamond,
    kernel_size: Annotated[int, typer.Option(..., "--kernel-size")] = 1,
    max_workers: Annotated[Optional[int], typer.Option(
        help="The maximum number of workers. If None, uses all available processors.",
    )] = 1,
):
    # create output directory
    out_dir.mkdir(exist_ok=True, parents=True)

    # land mask
    # this is the same landmask as the original IFT- can be downloaded w SOIT
    land_mask_ = create_land_mask(mask)

    # load files
    data_direc = in_dir
    ftci_direc = data_direc / "tci/"
    fcloud_direc = data_direc / "cloud/"

    # option to save figs after each step
    save_figs = save_figs

    ftcis = sorted(Path(ftci_direc).iterdir())
    fclouds = sorted(Path(fcloud_direc).iterdir())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ftci, fcloud in zip(ftcis, fclouds):
            future = executor.submit(
                preprocess,
                ftci,
                fcloud,
                land_mask_,
                itmax,
                itmin,
                step,
                kernel_type,
                kernel_size,
                save_figs,
                out_dir,
            )
            futures.append(future)

        # Wait for all threads to complete
        for future in futures:
            future.result()


if __name__ == "__main__":
    app()
