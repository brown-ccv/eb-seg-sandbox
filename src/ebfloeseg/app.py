#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import tomllib
import typer
from typing import Optional


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
epilog = f"Example: {name} --data-direc /path/to/data --save_figs --save-direc /path/to/save --land /path/to/landfile"
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


@app.command(name="process-images", help=help, epilog=epilog)
def process_images(
    config_file: Path = typer.Option(
        ...,
        "--config-file",
        "-c",
        help="Path to configuration file",
    ),
    max_workers: Optional[int] = typer.Option(
        None,
        help="The maximum number of workers. If None, uses all available processors.",
    ),
):

    args = parse_config_file(config_file)

    save_direc = args.save_direc

    # create output directory
    save_direc.mkdir(exist_ok=True, parents=True)

    # ## land mask
    # this is the same landmask as the original IFT- can be downloaded w SOIT
    land_mask = create_land_mask(args.land)

    # ## load files
    data_direc = args.data_direc
    ftci_direc = data_direc / "tci/"
    fcloud_direc = data_direc / "cloud/"

    # option to save figs after each step
    save_figs = args.save_figs

    ftcis = sorted(Path(ftci_direc).iterdir())
    fclouds = sorted(Path(fcloud_direc).iterdir())

    with ProcessPoolExecutor() as executor:
        futures = []
        for ftci, fcloud in zip(ftcis, fclouds):
            future = executor.submit(
                preprocess,
                ftci,
                fcloud,
                land_mask,
                args.itmax,
                args.itmin,
                args.step,
                args.kernel_type,
                args.kernel_size,
                save_figs,
                save_direc,
            )
            futures.append(future)

        # Wait for all threads to complete
        for future in futures:
            future.result()


if __name__ == "__main__":
    app()
