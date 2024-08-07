#!/usr/bin/env python

import logging
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
import pandas

from ebfloeseg.masking import create_land_mask
from ebfloeseg.preprocess import preprocess, preprocess_b
from ebfloeseg.io_ import load as load_
from ebfloeseg.io_ import ImageType, Satellite

_logger = logging.getLogger(__name__)

name = "fsdproc"
app = typer.Typer(name=name, add_completion=False)


@app.callback()
def main(quiet: Annotated[bool, typer.Option()] = False,
         verbose: Annotated[bool, typer.Option()] = False,
         debug: Annotated[bool, typer.Option()] = False,
         ):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    elif verbose:
        logging.basicConfig(level=logging.INFO)
    elif quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.WARNING)


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


@app.command(help="TODO: add description", epilog=f"Example: {name} --data-direc /path/to/data --save_figs --save-direc /path/to/save --land /path/to/landfile")
def process_batch(
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


@app.command()
def load(
    outfile: Annotated[Path, typer.Argument()],
    datetime: str = "2016-07-01T00:00:00Z",
    wrap: str = "day",
    satellite: Satellite = Satellite.terra,
    kind: ImageType = ImageType.truecolor,
    bbox: str = "-2334051.0214676396,-414387.78951688844,-1127689.8419350237,757861.8364224486",
    scale: Annotated[
        int, typer.Option(help="size of a pixel in units of the bounding box")
    ] = 250,
    crs: str = "EPSG:3413",
    ts: int = 1683675557694,
    format: str = "image/tiff",
):

    result = load_(
        datetime=datetime,
        wrap=wrap,
        satellite=satellite,
        kind=kind,
        bbox=bbox,
        scale=scale,
        crs=crs,
        ts=ts,
        format=format,
    )

    with open(outfile, "wb") as f:
        f.write(result["content"])

    return


class KernelType(str, Enum):
    diamond = "diamond"
    ellipse = "ellipse"


@app.command()
def process(
    truecolorimg: Annotated[Path, typer.Argument()],
    cloudimg: Annotated[Path, typer.Argument()],
    landmask: Annotated[Path, typer.Argument()],
    outdir: Annotated[Path, typer.Argument()],
    save_figs: Annotated[bool, typer.Option()] = True,
    out_prefix: Annotated[
        str, typer.Option(help="string to prepend to filenames")
    ] = "",
    itmax: Annotated[
        int,
        typer.Option(..., "--itmax",
                     help="maximum number of iterations for erosion"),
    ] = 8,
    itmin: Annotated[
        int,
        typer.Option(..., "--itmin",
                     help="minimum number of iterations for erosion"),
    ] = 3,
    step: Annotated[int, typer.Option(..., "--step")] = -1,
    kernel_type: Annotated[
        KernelType, typer.Option(..., "--kernel-type")
    ] = KernelType.diamond,
    kernel_size: Annotated[int, typer.Option(..., "--kernel-size")] = 1,
    date: Annotated[Optional[datetime], typer.Option()] = None,
):

    preprocess_b(
        ftci=truecolorimg,
        fcloud=cloudimg,
        fland=landmask,
        itmax=itmax,
        itmin=itmin,
        step=step,
        erosion_kernel_type=kernel_type,
        erosion_kernel_size=kernel_size,
        save_figs=save_figs,
        save_direc=outdir,
        fname_prefix=out_prefix,
        date=date,
    )

    return


@app.command()
def get_bbox(
    datafile: Annotated[Path, typer.Argument()],
    index: Annotated[str, typer.Argument()],
    index_col: Annotated[str, typer.Option()] = "location",
    colnames: Annotated[list[str], typer.Option()] = [
        "left_x",
        "lower_y",
        "right_x",
        "top_y",
    ],
    separator: Annotated[str, typer.Option()] = ",",
):

    df = pandas.read_csv(datafile, index_col=index_col)
    output = separator.join(str(s) for s in list(df.loc[index][colnames]))
    print(output)


if __name__ == "__main__":
    app()
