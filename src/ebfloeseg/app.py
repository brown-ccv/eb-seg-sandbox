#!/usr/bin/env python

from pathlib import Path
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor

import typer

from ebfloeseg.masking import create_land_mask
from ebfloeseg.process import process


def validate_kernel_type(ctx: typer.Context, value: str) -> str:
    if value not in ["diamond", "ellipse"]:
        raise typer.BadParameter("Kernel type must be 'diamond' or 'ellipse'")
    return value


help = "TODO: add description"
name = "fsdproc"
epilog = f"Example: {name} --data-direc /path/to/data --save_figs --save-direc /path/to/save --land /path/to/landfile"
app = typer.Typer(name=name, add_completion=False)


@app.command(name="process-images", help=help, epilog=epilog)
def process_images(
    data_direc: Path = typer.Option(..., help="directory containing the data"),
    save_figs: bool = typer.Option(False, help="whether to save figures"),
    save_direc: Path = typer.Option(..., help="directory to save figures"),
    land: Path = typer.Option(..., help="land mask to use"),
    erode_itmax: int = typer.Option(8, help="maximum number of iterations for erosion"),
    erode_itmin: int = typer.Option(
        3, help="(inclusive) minimum number of iterations for erosion"
    ),
    step: int = typer.Option(-1, help="step size for erosion"),
    erosion_kernel_type: str = typer.Option(
        "diamond",
        help="type of kernel (either diamond or ellipse)",
        callback=validate_kernel_type,
    ),
    erosion_kernel_size: int = typer.Option(1, help="size of the erosion kernel"),
):

    ftci_direc: Path = data_direc / "tci"
    fcloud_direc: Path = data_direc / "cloud"
    land_mask = create_land_mask(land)

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
            [save_figs] * m,
            [save_direc] * m,
            [land_mask] * m,
            [erode_itmax] * m,
            [erode_itmin] * m,
            [step] * m,
            [erosion_kernel_type] * m,
            [erosion_kernel_size] * m,
        )


if __name__ == "__main__":
    app()  # pragma: no cover
