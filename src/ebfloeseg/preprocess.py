import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import skimage
from skimage.filters import threshold_local
from skimage.morphology import diamond, opening, dilation, binary_dilation

from ebfloeseg.masking import maskrgb, mask_image
from ebfloeseg.savefigs import imsave, save_ice_mask_hist
from ebfloeseg.utils import write_mask_values, get_wcuts

def get_remove_small_mask(watershed, it):
    area_lim = (it) ** 4
    props = skimage.measure.regionprops_table(
        watershed, properties=["label", "area"]
    )
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


def preprocess(
    tci,
    cloud_mask,
    land_mask,
    erosion_kernel_type,
    erosion_kernel_size,
    erode_itmax,
    erode_itmin,
    step,
    save_figs,
    target_dir,
    doy,
    year,
):
    # Reshape tci to RGB (3, x, y) => (x, y, 3)
    red_c, green_c, blue_c = tci.read()
    rgb_masked = np.dstack([red_c, green_c, blue_c])  # masked below

    maskrgb(rgb_masked, cloud_mask)
    if save_figs:
        imsave(tci, rgb_masked, target_dir, doy, "cloud_mask_on_rgb.tif")

    maskrgb(rgb_masked, land_mask)
    if save_figs:
        imsave(tci, rgb_masked, target_dir, doy, "land_cloud_mask_on_rgb.tif")

    ## adaptive threshold for ice mask
    red_masked = rgb_masked[:, :, 0]
    thresh_adaptive = threshold_local(red_c, block_size=399)

    # here just determining the min and max values for the adaptive threshold
    ow_cut_min, ow_cut_max, bins = get_wcuts(red_masked)

    if save_figs:
        save_ice_mask_hist(red_masked, bins, ow_cut_min, ow_cut_max, doy, target_dir)

    # clamp thresh_adaptive
    thresh_adaptive = np.clip(thresh_adaptive, ow_cut_min, ow_cut_max)

    ice_mask = red_masked > thresh_adaptive

    land_cloud_mask = land_mask + cloud_mask
    write_mask_values(
        land_mask, land_cloud_mask, ice_mask, doy, year, target_dir
    )  # Should this be done conditionally? CP

    # saving ice mask
    if save_figs:
        imsave(
            tci,
            ice_mask,
            target_dir,
            doy,
            "ice_mask_bw.tif",
            count=1,
            rollaxis=False,
            as_uint8=True,
        )

    # here dilating the land and cloud mask so any floes that are adjacent to the mask can be removed later
    land_cloud_mask_dilated = binary_dilation(land_cloud_mask.astype(int), diamond(10))

    # TODO: move to a function erosion_expansion_algo
    # setting up different kernel for erosion-expansion algo
    erosion_kernel = get_erosion_kernel(erosion_kernel_type, erosion_kernel_size)

    output = np.zeros(np.shape(ice_mask))
    inpuint8 = ice_mask.astype(np.uint8)

    # Test refactoring
    inp = ice_mask
    _output = np.zeros(np.shape(ice_mask))

    for r, it in enumerate(range(erode_itmax, erode_itmin - 1, step)):
        # erode a lot at first, decrease number of iterations each time
        eroded_ice_mask = cv2.erode(inpuint8, erosion_kernel, iterations=it).astype(
            np.uint8
        )
        eroded_ice_mask = ndimage.binary_fill_holes(eroded_ice_mask).astype(np.uint8)

        dilated_ice_mask = cv2.dilate(inpuint8, erosion_kernel, iterations=it).astype(
            np.uint8
        )

        # label floes remaining after erosion
        ret, markers = cv2.connectedComponents(eroded_ice_mask)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        unknown = cv2.subtract(dilated_ice_mask, eroded_ice_mask)

        # Now, mark the region of unknown with zero
        mask_image(markers, unknown == 255, 0)

        # dilate each marker
        for _ in np.arange(0, it + 1):
            markers = dilation(markers, erosion_kernel)

        # rewatershed
        watershed = cv2.watershed(rgb_masked, markers)

        # get rid of floes that intersect the dilated land mask (WARNING: cloud mask already included)
        condition = np.isin(
            watershed, np.unique(watershed[land_cloud_mask_dilated & (watershed > 1)])
        )
        mask_image(watershed, condition, 1)

        # set the open water and already identified floes to no
        # watershed[~ice_mask] = 1
        mask_image(watershed, ~ice_mask, 1)

        # get rid of ones that are too small
        mask = get_remove_small_mask(watershed, it)
        mask_image(watershed, mask, 1)

        if save_figs:
            fname = f"identification_round_{r}.tif"
            imsave(
                tci,
                watershed,
                target_dir,
                doy,
                fname,
                count=1,
                rollaxis=False,
                as_uint8=True,
            )

        mask_image(watershed, watershed < 2, 0)
        output += watershed

    output = opening(output)

    return output, red_c
