from enum import Enum
import logging
from pathlib import Path
from typing import Annotated
import requests

import typer

_logger = logging.getLogger(__name__)

class ImageType(str, Enum):
    truecolor = "truecolor"
    cloud = "cloud"
    landmask = "landmask"

def get_width_height(bbox: str, scale: float):
    """Get width and height for a bounding box where one pixel corresponds to `scale` bounding box units
    
    Examples:
        >>> get_width_height("0,0,1,1", 10)
        (10, 10)

        >>> get_width_height("0,0,1,5", 10)
        (2, 10)

        >>> get_width_height("0,0,1,5", 10)
        (2, 10)

    """
    x1, y1, x2, y2 = [float(n) for n in bbox.split(",")]
    x_length = abs(x2 - x1)
    y_length = abs(y2 - y1)

    width, height = int(x_length / scale), int(y_length / scale)
    return width, height

app = typer.Typer()


@app.command()
def main(
    outfile: Annotated[Path, typer.Argument()],
    datetime: str = "2016-07-01T00:00:00Z",
    wrap: str = "day",
    kind: ImageType = ImageType.truecolor,
    bbox: str = "-2334051.0214676396,-414387.78951688844,-1127689.8419350237,757861.8364224486",
    scale: Annotated[int, typer.Option(help="size of a pixel in units of the bounding box")] = 250,
    crs: str = "EPSG:3413",
    ts: int = 1683675557694,
    format: str = "image/tiff",
    quiet: Annotated[bool, typer.Option()]=False,
    verbose: Annotated[bool, typer.Option()]=False,
    debug: Annotated[bool, typer.Option()]=False,
):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    elif verbose:
        logging.basicConfig(level=logging.INFO)
    elif quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.WARNING)

    match kind:
        case ImageType.truecolor:
            layers = "MODIS_Terra_CorrectedReflectance_TrueColor"
        case ImageType.cloud:
            layers = "MODIS_Terra_Cloud_Fraction_Day"
        case ImageType.landmask:
            layers = "OSM_Land_Mask"


    width, height = get_width_height(bbox, scale)
    _logger.info("Width: %s Height: %s" % (width, height))

    url = f"https://wvs.earthdata.nasa.gov/api/v1/snapshot"
    payload = {
        "REQUEST": "GetSnapshot",
        "TIME": datetime,
        "BBOX": bbox,
        "CRS": crs,
        "LAYERS": layers,
        "WRAP": wrap,
        "FORMAT": format,
        "WIDTH": width,
        "HEIGHT": height,
        "ts": ts
    }
    r = requests.get(url, params=payload, allow_redirects=True)
    r.raise_for_status()


    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "wb") as f:
        f.write(r.content)

    return


if __name__ == "__main__":
    app()
