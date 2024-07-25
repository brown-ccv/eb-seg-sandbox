import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import skimage
from skimage.filters import threshold_local
from skimage.morphology import diamond, opening, dilation, binary_dilation

from ebfloeseg.masking import maskrgb, mask_image
from ebfloeseg.savefigs import imsave
from ebfloeseg.utils import write_mask_values
from ebfloeseg.peakdet import peakdet


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
    rgb = np.dstack([red_c, green_c, blue_c])

    maskrgb(rgb, cloud_mask)
    if save_figs:
        imsave(tci, rgb, target_dir, doy, "cloud_mask_on_rgb.tif")

    maskrgb(rgb, land_mask)
    if save_figs:
        imsave(tci, rgb, target_dir, doy, "land_cloud_mask_on_rgb.tif")

    ## adaptive threshold for ice mask
    thresh_adaptive = threshold_local(red_c, block_size=399)
    image = red_masked = rgb[:, :, 0]

    # here just determining the min and max values for the adaptive threshold
    binz = np.arange(1, 256, 5)
    rn, rbins = np.histogram(red_masked.flatten(), bins=binz)
    dx = 0.01 * np.mean(rn)
    rmaxtab, rmintab = peakdet(rn, dx)
    rmax_n = rbins[rmaxtab[-1, 0]]
    rhm_high = rmaxtab[-1, 1] / 2

    ow_cut_min = 100 if ~np.any(rmintab) else rbins[rmintab[-1, 0]]

    ow_cut_max_cond = np.where(
        (rbins[:-1] < rmax_n) & (rn <= rhm_high)
    )  # TODO: add comment
    if np.any(ow_cut_max_cond):
        ow_cut_max = rbins[ow_cut_max_cond[0][-1]]  # fwhm to left of ice max
    else:
        ow_cut_max = rmax_n - 10

    if save_figs:
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        plt.hist(red_masked.flatten(), bins=binz, color="r")
        plt.axvline(ow_cut_max)
        plt.axvline(ow_cut_min)
        plt.savefig(target_dir / "ice_mask_hist.png")

    # mask thresh_adaptive
    mask_image(thresh_adaptive, thresh_adaptive < ow_cut_min, ow_cut_min)
    mask_image(thresh_adaptive, thresh_adaptive > ow_cut_max, ow_cut_max)

    ice_mask = image > thresh_adaptive
    lmd = land_mask + cloud_mask

    write_mask_values(
        land_mask, lmd, ice_mask, doy, year, target_dir
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
    lmd = binary_dilation(lmd.astype(int), diamond(10))

    # setting up different kernel for erosion-expansion algo
    if erosion_kernel_type == "diamond":
        kernel_er = diamond(erosion_kernel_size)
    elif erosion_kernel_type == "ellipse":
        kernel_er = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple([erosion_kernel_size] * 2)
        )

    output = np.zeros(np.shape(ice_mask))
    inpuint8 = ice_mask.astype(np.uint8)

    for r, it in enumerate(range(erode_itmax, erode_itmin - 1, step)):
        # erode a lot at first, decrease number of iterations each time
        eroded_ice_mask = cv2.erode(inpuint8, kernel_er, iterations=it).astype(np.uint8)
        eroded_ice_mask = ndimage.binary_fill_holes(eroded_ice_mask).astype(np.uint8)

        dilated_ice_mask = cv2.dilate(inpuint8, kernel_er, iterations=it).astype(
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
            markers = dilation(markers, kernel_er)

        # rewatershed
        watershed = cv2.watershed(rgb, markers)

        # get rid of floes that intersect the dilated land mask
        watershed[np.isin(watershed, np.unique(watershed[(lmd) & (watershed > 1)]))] = 1

        # set the open water and already identified floes to no
        watershed[~ice_mask] = 1

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
                target_dir,
                doy,
                fname,
                count=1,
                rollaxis=False,
                as_uint8=True,
            )

        watershed[watershed < 2] = 0
        output = watershed + output

    output = opening(output)

    return output, red_c