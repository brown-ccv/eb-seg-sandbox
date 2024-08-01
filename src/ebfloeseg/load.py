from enum import Enum
from pathlib import Path
from typing import Annotated
import requests

import typer


class ImageType(str, Enum):
    truecolor = "truecolor"
    cloud = "cloud"
    landmask = "landmask"


app = typer.Typer()


@app.command()
def main(
    outfile: Annotated[Path, typer.Argument()],
    datetime: str = "2016-07-01T00:00:00Z",
    wrap: str = "day",
    kind: ImageType = ImageType.truecolor,
    bbox: str = "-2334051.0214676396,-414387.78951688844,-1127689.8419350237,757861.8364224486",
    width: int = 4712,
    height: int = 4579,
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

    

    url = f"https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME={datetime}&BBOX={bbox}&CRS={crs}&LAYERS={layers}&WRAP={wrap}&FORMAT={format}&WIDTH={width}&HEIGHT={height}&ts={ts}"

    r = requests.get(url, allow_redirects=True)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "wb") as f:
        f.write(r.content)

    return


if __name__ == "__main__":
    app()
