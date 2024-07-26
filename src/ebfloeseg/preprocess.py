from logging import getLogger

import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
import skimage
from skimage.filters import threshold_local
from skimage.morphology import diamond, opening
import rasterio

from ebfloeseg.masking import maskrgb, mask_image, create_cloud_mask
from ebfloeseg.savefigs import imsave, save_ice_mask_hist
from ebfloeseg.utils import (
    write_mask_values,
    get_wcuts,
    getmeta,
    getres,
    get_region_properties,
)


def extract_features(
    output, red_c, target_dir, res, sat, doy
):  # adding doy temporarily for testing. TODO: use doy for subdir
    # fname = target_dir / f"{res}_{sat}_props.csv"
    fname = target_dir / f"{res}_{sat}_{doy}_props.csv"
    props = get_region_properties(output, red_c)
    df = pd.DataFrame.from_dict(props)
    df.to_csv(fname)


def get_remove_small_mask(watershed, it):
    area_lim = (it) ** 4
    props = skimage.measure.regionprops_table(watershed, properties=["label", "area"])
    df = pd.DataFrame.from_dict(props)
    mask = np.isin(watershed, df[df.area < area_lim].label.values)
    return mask


def get_erosion_kernel(erosion_kernel_type="diamond", erosion_kernel_size=1):
    if erosion_kernel_type == "diamond":
        erosion_kernel = diamond(erosion_kernel_size)
    elif erosion_kernel_type == "ellipse":
        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple([erosion_kernel_size] * 2)
        )
    return erosion_kernel


logger = getLogger(__name__)


def _preprocess(
    ftci,
    fcloud,
    land_mask,
    itmax,
    itmin,
    step,
    erosion_kernel_type,
    erosion_kernel_size,
    save_figs,
    save_direc,
):
    tci = rasterio.open(ftci)
    doy, year, sat = getmeta(fcloud)
    res = getres(doy, year)

    cloud_mask = create_cloud_mask(fcloud)

    red_c, green_c, blue_c = tci.read()  # these are different than the layers in rgb
    rgb_masked = np.dstack([red_c, green_c, blue_c])  # masked below

    maskrgb(rgb_masked, cloud_mask)
    if save_figs:
        fname = f"{doy}_cloud_mask_on_rgb.tif"
        imsave(tci, rgb_masked, save_direc, doy, fname)

    maskrgb(rgb_masked, land_mask)
    if save_figs:
        fname = f"{doy}_land_cloud_mask_on_rgb.tif"
        imsave(tci, rgb_masked, save_direc, doy, fname)

    ## adaptive threshold for ice mask
    red_masked = rgb_masked[:, :, 0]
    thresh_adaptive = threshold_local(red_c, block_size=399)

    # here just determining the min and max values for the adaptive threshold
    ow_cut_min, ow_cut_max, bins = get_wcuts(red_masked)

    if save_figs:
        save_ice_mask_hist(red_masked, bins, ow_cut_min, ow_cut_max, doy, save_direc)

    thresh_adaptive = np.clip(thresh_adaptive, ow_cut_min, ow_cut_max)

    ice_mask = red_masked > thresh_adaptive

    # a simple text file with columns: 'doy','ice_area','unmasked','sic'
    write_mask_values(
        land_mask, land_mask + cloud_mask, ice_mask, doy, year, save_direc
    )

    # saving ice mask
    if save_figs:
        imsave(
            tci,
            ice_mask,
            save_direc,
            doy,
            "ice_mask_bw.tif",
            count=1,
            rollaxis=False,
            as_uint8=True,
            res=res,
        )

    # here dilating the land and cloud mask so any floes that are adjacent to the mask can be removed later
    land_cloud_mask = (land_mask + cloud_mask).astype(int)
    land_cloud_mask_dilated = skimage.morphology.binary_dilation(
        land_cloud_mask, diamond(10)
    )

    # setting up different kernel for erosion-expansion algo
    erosion_kernel = get_erosion_kernel(erosion_kernel_type, erosion_kernel_size)

    # TODO: clarify this block
    inp = ice_mask
    input_no = ice_mask
    output = np.zeros((np.shape(ice_mask)))
    itmax = itmax
    itmin = itmin
    step = step
    for r, it in enumerate(range(itmax, itmin - 1, step)):
        # erode a lot at first, decrease number of iterations each time
        eroded_ice_mask = cv2.erode(inp.astype(np.uint8), erosion_kernel, iterations=it)
        eroded_ice_mask = ndimage.binary_fill_holes(eroded_ice_mask.astype(np.uint8))

        dilated_ice_mask = cv2.dilate(
            inp.astype(np.uint8), erosion_kernel, iterations=it
        )

        # label floes remaining after erosion
        ret, markers = cv2.connectedComponents(eroded_ice_mask.astype(np.uint8))

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        unknown = cv2.subtract(
            dilated_ice_mask.astype(np.uint8), eroded_ice_mask.astype(np.uint8)
        )

        # Now, mark the region of unknown with zero
        # markers[unknown == 255] = 0
        mask_image(markers, unknown == 255, 0)

        # dilate each marker
        for a in np.arange(0, it + 1, 1):
            markers = skimage.morphology.dilation(markers, erosion_kernel)

        # rewatershed
        watershed = cv2.watershed(rgb_masked, markers)

        # get rid of floes that intersect the dilated land mask
        watershed[
            np.isin(
                watershed,
                np.unique(watershed[land_cloud_mask_dilated & (watershed > 1)]),
            )
        ] = 1

        # pdb.set_trace()
        # set the open water and already identified floes to no
        # watershed[~input_no] = 1
        mask_image(watershed, ~input_no, 1)

        # get rid of ones that are too small
        area_lim = (it) ** 4
        props = skimage.measure.regionprops_table(
            watershed, properties=["label", "area"]
        )
        df = pd.DataFrame.from_dict(props)
        watershed[np.isin(watershed, df[df.area < area_lim].label.values)] = 1

        if save_figs:
            fname = f"identification_round_{r}.tif"
            imsave(
                tci,
                watershed,
                save_direc,
                doy,
                fname,
                count=1,
                rollaxis=False,
                as_uint8=True,
                res=res,
            )

        input_no = ice_mask + inp
        inp = (watershed == 1) & (inp == 1) & ice_mask
        watershed[watershed < 2] = 0
        output += watershed

    # saving the props table
    output = opening(output)
    extract_features(output, red_c, save_direc, res, sat, doy)

    # saving the label floes tif
    fname = f"{sat}_final.tif"
    imsave(
        tci,
        output,
        save_direc,
        doy,
        fname,
        count=1,
        rollaxis=False,
        as_uint8=True,
        res=res,
    )


def preprocess(
    ftci,
    fcloud,
    land_mask,
    itmax,
    itmin,
    step,
    erosion_kernel_type,
    erosion_kernel_size,
    save_figs,
    save_direc,
):
    try:
        _preprocess(
            ftci,
            fcloud,
            land_mask,
            itmax,
            itmin,
            step,
            erosion_kernel_type,
            erosion_kernel_size,
            save_figs,
            save_direc,
        )
    except Exception as e:
        logger.exception(f"Error processing {fcloud} and {ftci}: {e}")
        raise
