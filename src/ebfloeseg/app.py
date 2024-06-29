#!/usr/bin/env python

from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor


import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from skimage.morphology import diamond, opening, dilation, binary_dilation
from scipy import ndimage
import skimage
from skimage.filters import threshold_local
import cv2
import typer

from ebfloeseg.peakdet import peakdet
from ebfloeseg.utils import getmeta, getres, write_mask_values, get_region_properties
from ebfloeseg.masking import maskrgb, create_cloud_mask, create_land_mask, mask_image
from ebfloeseg.savefigs import imsave


def process(
    fcloud,
    ftci,
    fcloud_direc,
    ftci_direc,
    save_figs,
    save_direc,
    land_mask,
    erode_itmax: int = 8,
    erode_itmin: int = 3,
    step: int = -1,
    erosion_kernel_type: str = "diamond",
    erosion_kernel_size: int = 1,
):

    doy, year, sat = getmeta(fcloud)
    res = getres(doy, year)

    target_dir = Path(save_direc, doy)
    target_dir.mkdir(exist_ok=True, parents=True)

    # open all files
    tci = rasterio.open(ftci_direc / ftci)
    cloud_mask = create_cloud_mask(fcloud_direc / fcloud)

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
    write_mask_values(land_mask, lmd, ice_mask, doy, year, target_dir)

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

    # here dialating the land and cloud mask so any floes that are adjacent to the mask can be removed later
    lmd = binary_dilation(lmd.astype(int), diamond(10))

    # setting up different kernel for erosion-expansion algo
    if erosion_kernel_type == "diamond":
        kernel_er = diamond(erosion_kernel_size)
    elif erosion_kernel_type == "ellipse":
        kernel_er = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple([erosion_kernel_size] * 2)
        )

    inp = ice_mask
    input_no = ice_mask
    output = np.zeros(np.shape(ice_mask))
    inpuint8 = inp.astype(np.uint8)

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
        watershed[~input_no] = 1

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

        inp = (watershed == 1) & (inp == 1) & ice_mask
        watershed[watershed < 2] = 0
        output = watershed + output

    # saving the props table and label floes tif
    output = opening(output)
    props = get_region_properties(output, red_c)

    df = pd.DataFrame.from_dict(props)
    fname = target_dir / f"{res}_{sat}_props.csv"
    df.to_csv(fname)

    # saving the label floes tif
    fname = f"{sat}_final.tif"
    imsave(
        tci,
        output,
        target_dir,
        doy,
        fname,
        count=1,
        rollaxis=False,
        as_uint8=True,
        res=res,
    )


def validate_kernel_type(ctx: typer.Context, value: str) -> str:
    if value not in ["diamond", "ellipse"]:
        raise typer.BadParameter("Kernel type must be 'diamond' or 'ellipse'")
    return value


help = "TODO: add description"
name = "ebprog"
epilog = "Example: ebprog process_images --data_direc /path/to/data --save_figs --save_direc /path/to/save --land /path/to/landmask"
app = typer.Typer(name=name, help=help, epilog=epilog, add_completion=False)


@app.command(name="process_images")
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
    app()
