from collections import namedtuple
import io
import logging
from enum import Enum

import rasterio
import requests
import numpy as np
from rasterio.enums import ColorInterp

_logger = logging.getLogger(__name__)


class ImageType(str, Enum):
    truecolor = "truecolor"
    cloud = "cloud"
    landmask = "landmask"


class Satellite(str, Enum):
    terra = "terra"
    aqua = "aqua"


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


def image_not_empty(img: rasterio.DatasetReader):
    # check that the image isn't all zeros using img.read() and the .colorinterp field
    match img.colorinterp:
        case (ColorInterp.red, ColorInterp.green, ColorInterp.blue):
            red_c, green_c, blue_c = img.read()
            return np.any(red_c) or np.any(green_c) or np.any(blue_c)
        case (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha):
            red_c, green_c, blue_c, alpha_c = img.read()
            return (np.any(red_c) or np.any(green_c) or np.any(blue_c)) and np.any(
                alpha_c
            )
        case _:
            msg = "unknown dimensions %s" % img.colorinterp
            raise ValueError(msg)


LoadResult = namedtuple("LoadResult", ["content", "img"])


def load(
    datetime: str = "2016-07-01T00:00:00Z",
    wrap: str = "day",
    satellite: Satellite = Satellite.terra,
    kind: ImageType = ImageType.truecolor,
    bbox: str = "-2334051.0214676396,-414387.78951688844,-1127689.8419350237,757861.8364224486",
    scale: int = 250,
    crs: str = "EPSG:3413",
    ts: int = 1683675557694,
    format: str = "image/tiff",
    validate: bool = True,
) -> LoadResult:

    match (satellite, kind):
        case (Satellite.terra, ImageType.truecolor):
            layers = "MODIS_Terra_CorrectedReflectance_TrueColor"
        case (Satellite.terra, ImageType.cloud):
            layers = "MODIS_Terra_Cloud_Fraction_Day"
        case (Satellite.aqua, ImageType.truecolor):
            layers = "MODIS_Aqua_CorrectedReflectance_TrueColor"
        case (Satellite.aqua, ImageType.cloud):
            layers = "MODIS_Aqua_Cloud_Fraction_Day"
        case (_, ImageType.landmask):
            layers = "OSM_Land_Mask"
        case _:
            msg = "satellite=%s and image kind=%s not supported" % (satellite, kind)
            raise NotImplementedError(msg)

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
        "ts": ts,
    }
    r = requests.get(url, params=payload, allow_redirects=True)
    r.raise_for_status()

    img = rasterio.open(io.BytesIO(r.content))

    if validate:
        assert image_not_empty(img)

    return LoadResult(r.content, img)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    load(kind=ImageType.truecolor)
    load(kind=ImageType.cloud)
    load(kind=ImageType.landmask)
