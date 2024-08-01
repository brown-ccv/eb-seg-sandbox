from enum import Enum
from pathlib import Path
from typing import Annotated
import requests

import typer


class ImageType(str, Enum):
    truecolor = "truecolor"
    cloud = "cloud"
    landmask = "landmask"

def get_width_height(bbox: str, size: int):
    """Get width and height for a bounding box where the longer axis has length `size`
    
    Examples:
        >>> get_width_height("0,0,1,1", 10)
        (10, 10)

        >>> get_width_height("0,0,1,5", 10)
        (2, 10)

        >>> get_width_height("0,0,1,5", 10)
        (2, 10)

    """
    x1, y1, x2, y2 = [float(n) for n in bbox.split(",")]
    x_size = abs(x2 - x1)
    y_size = abs(y2 - y1)

    max_size = max(x_size, y_size)

    width, height = int(size * x_size / max_size), int(size * y_size / max_size)
    return width, height

app = typer.Typer()


@app.command()
def main(
    outfile: Annotated[Path, typer.Argument()],
    datetime: str = "2016-07-01T00:00:00Z",
    wrap: str = "day",
    kind: ImageType = ImageType.truecolor,
    bbox: str = "-2334051.0214676396,-414387.78951688844,-1127689.8419350237,757861.8364224486",
    size: int = 4096,
    crs: str = "EPSG:3413",
    ts: int = 1683675557694,
    format: str = "image/tiff",
):

    match kind:
        case ImageType.truecolor:
            layers = "MODIS_Terra_CorrectedReflectance_TrueColor"
        case ImageType.cloud:
            layers = "MODIS_Terra_Cloud_Fraction_Day"
        case ImageType.landmask:
            layers = "OSM_Land_Mask"


    width, height = get_width_height(bbox, size)
    print(width, height)

    url = f"https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME={datetime}&BBOX={bbox}&CRS={crs}&LAYERS={layers}&WRAP={wrap}&FORMAT={format}&WIDTH={width}&HEIGHT={height}&ts={ts}"

    r = requests.get(url, allow_redirects=True)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "wb") as f:
        f.write(r.content)

    return


if __name__ == "__main__":
    app()
