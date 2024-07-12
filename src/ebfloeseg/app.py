#!/usr/bin/env python

from pathlib import Path
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor

import typer

from ebfloeseg.masking import create_land_mask
from ebfloeseg.process import process
from dataclasses import dataclass


@dataclass
class ConfigParams:
    data_direc: Path
    land: Path
    save_figs: bool
    save_direc: Path
    erode_itmax: int
    erode_itmin: int
    step: int
    erosion_kernel_type: str
    erosion_kernel_size: int


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
        "erode_itmax": 8,  # maximum number of iterations for erosion
        "erode_itmin": 3,  # (inclusive) minimum number of iterations for erosion
        "step": -1,
        "erosion_kernel_type": "diamond",  # type of kernel (either diamond or ellipse)
        "erosion_kernel_size": 1,
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
):

    params = parse_config_file(config_file)
    data_direc = params.data_direc

    ftci_direc: Path = data_direc / "tci"
    fcloud_direc: Path = data_direc / "cloud"
    land_mask = create_land_mask(params.land)

    ftcis = sorted(os.listdir(ftci_direc))
    fclouds = sorted(os.listdir(fcloud_direc))
    m = len(fclouds)

    with ProcessPoolExecutor() as executor:
        executor.map(
            process,
            fclouds,
            ftcis,
            [fcloud_direc] * m,
            [ftci_direc] * m,
            [params.save_figs] * m,
            [params.save_direc] * m,
            [land_mask] * m,
            [params.erode_itmax] * m,
            [params.erode_itmin] * m,
            [params.step] * m,
            [params.erosion_kernel_type] * m,
            [params.erosion_kernel_size] * m,
        )


if __name__ == "__main__":
    app()  # pragma: no cover
