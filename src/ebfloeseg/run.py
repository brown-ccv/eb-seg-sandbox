from enum import Enum
import logging
from pathlib import Path
from typing import Annotated, Optional
import requests

import typer

from ebfloeseg.load import logger_config
from ebfloeseg.masking import create_land_mask
from ebfloeseg.preprocess import preprocess_b

_logger = logging.getLogger(__name__)

from datetime import datetime


class KernelType(str, Enum):
    diamond = "diamond"
    ellipse = "ellipse"


app = typer.Typer()


@app.command()
def main(
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
        typer.Option(..., "--itmax", help="maximum number of iterations for erosion"),
    ] = 8,
    itmin: Annotated[
        int,
        typer.Option(..., "--itmin", help="minimum number of iterations for erosion"),
    ] = 3,
    step: Annotated[int, typer.Option(..., "--step")] = -1,
    kernel_type: Annotated[
        KernelType, typer.Option(..., "--kernel-type")
    ] = KernelType.diamond,
    kernel_size: Annotated[int, typer.Option(..., "--kernel-size")] = 1,
    date: Annotated[Optional[datetime], typer.Option()] = None,
    quiet: Annotated[bool, typer.Option()] = False,
    verbose: Annotated[bool, typer.Option()] = False,
    debug: Annotated[bool, typer.Option()] = False,
):

    logger_config(debug, verbose, quiet)

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


if __name__ == "__main__":
    app()
