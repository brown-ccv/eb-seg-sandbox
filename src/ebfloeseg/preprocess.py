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
from ebfloeseg.peakdet import peakdet, _peakdet


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

    # Test refactoring
    _red_cmasked = np.copy(tci.read()[0])
    _green_cmasked = np.copy(tci.read()[1])
    _blue_cmasked = np.copy(tci.read()[2])
    _red_cmasked[cloud_mask] = 0
    _green_cmasked[cloud_mask] = 0
    _blue_cmasked[cloud_mask] = 0
    _rgb_cloudmasked = np.dstack([_red_cmasked, _green_cmasked, _blue_cmasked])
    assert np.array_equal(rgb, _rgb_cloudmasked)

    if save_figs:
        imsave(tci, rgb, target_dir, doy, "cloud_mask_on_rgb.tif")

    maskrgb(rgb, land_mask)

    # Test refactoring
    _red_cmasked[land_mask] = 0
    _green_cmasked[land_mask] = 0
    _blue_cmasked[land_mask] = 0
    _rgb_cloudmasked_landmasked = np.dstack(
        [_red_cmasked, _green_cmasked, _blue_cmasked]
    )
    assert np.array_equal(rgb, _rgb_cloudmasked_landmasked)

    if save_figs:
        imsave(tci, rgb, target_dir, doy, "land_cloud_mask_on_rgb.tif")

    ## adaptive threshold for ice mask
    thresh_adaptive = threshold_local(red_c, block_size=399)
    image = red_masked = rgb[:, :, 0]

    # Test refactoring
    _red_masked = _red_c = tci.read()[0]
    assert np.array_equal(_red_c, _red_masked)
    _image = _red_masked
    assert np.array_equal(red_c, _image)
    _thresh_adaptive = threshold_local(_image, block_size=399)
    assert np.array_equal(thresh_adaptive, _thresh_adaptive)
    _red_masked[(land_mask | cloud_mask)] = 0
    assert np.array_equal(_red_masked, red_masked)
    # assert False

    # here just determining the min and max values for the adaptive threshold
    binz = np.arange(1, 256, 5)
    rn, rbins = np.histogram(red_masked.flatten(), bins=binz)
    dx = 0.01 * np.mean(rn)
    rmaxtab, rmintab = peakdet(rn, dx)

    # Test refactoring
    _rmaxtab, _rmintab = _peakdet(rn, dx, x=None)
    assert np.array_equal(rmaxtab, _rmaxtab)
    assert np.array_equal(rmintab, _rmintab)
    # assert False

    rmax_n = rbins[rmaxtab[-1, 0]]
    rhm_high = rmaxtab[-1, 1] / 2

    # TODO: move to a function get_ow_cut_min_max
    ow_cut_min = 100 if ~np.any(rmintab) else rbins[rmintab[-1, 0]]
    ow_cut_max_cond = np.where(
        (rbins[:-1] < rmax_n) & (rn <= rhm_high)
    )  # TODO: add comment
    if np.any(ow_cut_max_cond):
        ow_cut_max = rbins[ow_cut_max_cond[0][-1]]  # fwhm to left of ice max
    else:
        ow_cut_max = rmax_n - 10

    # Test refactoring
    if ~np.any(rmintab):
        _ow_cut_min = 100
    else:
        _ow_cut_min = rbins[rmintab[-1, 0]]
    if np.any(np.where((rbins[:-1] < rmax_n) & (rn <= rhm_high))):
        _ow_cut_max = rbins[
            np.where((rbins[:-1] < rmax_n) & (rn <= rhm_high))[0][-1]
        ]  # fwhm to left of ice max
    else:
        _ow_cut_max = rmax_n - 10
    assert ow_cut_min == _ow_cut_min
    assert ow_cut_max == _ow_cut_max
    # assert False

    if save_figs:
        # TODO: move to a function save_ice_mask_hist
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        plt.hist(red_masked.flatten(), bins=binz, color="r")
        plt.axvline(ow_cut_max)
        plt.axvline(ow_cut_min)
        plt.savefig(target_dir / "ice_mask_hist.png")

    # mask thresh_adaptive
    mask_image(thresh_adaptive, thresh_adaptive < ow_cut_min, ow_cut_min)
    mask_image(thresh_adaptive, thresh_adaptive > ow_cut_max, ow_cut_max)

    # Test refactoring
    _thresh_adaptive[_thresh_adaptive < ow_cut_min] = _ow_cut_min
    _thresh_adaptive[_thresh_adaptive > ow_cut_max] = _ow_cut_max
    assert np.array_equal(thresh_adaptive, _thresh_adaptive)
    # assert False

    ice_mask = image > thresh_adaptive

    # Test refactoring
    _ice_mask = _image > _thresh_adaptive
    assert np.array_equal(ice_mask, _ice_mask)
    # assert False

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

    # Test refactoring
    kernel = diamond(10)
    _lmd = binary_dilation((land_mask + cloud_mask).astype(int), kernel)
    assert np.array_equal(lmd, _lmd)
    # assert False

    # TODO: move to a function erosion_expansion_algo
    # setting up different kernel for erosion-expansion algo
    if erosion_kernel_type == "diamond":
        kernel_er = diamond(erosion_kernel_size)
    elif erosion_kernel_type == "ellipse":
        kernel_er = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple([erosion_kernel_size] * 2)
        )

    output = np.zeros(np.shape(ice_mask))
    inpuint8 = ice_mask.astype(np.uint8)

    # Test refactoring
    inp = ice_mask
    _output = np.zeros(np.shape(ice_mask))

    for r, it in enumerate(range(erode_itmax, erode_itmin - 1, step)):
        # erode a lot at first, decrease number of iterations each time
        eroded_ice_mask = cv2.erode(inpuint8, kernel_er, iterations=it).astype(np.uint8)
        eroded_ice_mask = ndimage.binary_fill_holes(eroded_ice_mask).astype(np.uint8)

        # Test refactoring
        im3 = cv2.erode(inp.astype(np.uint8), kernel_er, iterations=it)
        im3 = ndimage.binary_fill_holes(im3.astype(np.uint8))
        assert np.array_equal(eroded_ice_mask, im3)
        # assert False

        dilated_ice_mask = cv2.dilate(inpuint8, kernel_er, iterations=it).astype(
            np.uint8
        )

        # Test refactoring
        im2 = cv2.dilate(inp.astype(np.uint8), kernel_er, iterations=it)
        assert np.array_equal(dilated_ice_mask, im2)
        # assert False

        # label floes remaining after erosion
        ret, markers = cv2.connectedComponents(eroded_ice_mask)

        # Test refactoring
        _ret, _markers = cv2.connectedComponents(im3.astype(np.uint8))
        assert ret == _ret
        assert np.array_equal(markers, _markers)
        # assert False

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        unknown = cv2.subtract(dilated_ice_mask, eroded_ice_mask)

        # Test refactoring
        _markers = _markers + 1
        _unknown = cv2.subtract(im2.astype(np.uint8), im3.astype(np.uint8))
        assert np.array_equal(unknown, _unknown)
        # assert False

        # Now, mark the region of unknown with zero
        mask_image(markers, unknown == 255, 0)

        # Test refactoring
        _markers[_unknown == 255] = 0
        assert np.array_equal(markers, _markers)
        # assert False

        # dilate each marker
        for _ in np.arange(0, it + 1):
            markers = dilation(markers, kernel_er)

        # Test refactoring
        for a in np.arange(0, it + 1, 1):
            _markers = dilation(_markers, kernel_er)
        assert np.array_equal(markers, _markers)
        # assert False

        # rewatershed
        watershed = cv2.watershed(rgb, markers)

        # Test refactoring
        im4 = cv2.watershed(rgb, _markers)
        assert np.array_equal(watershed, im4)
        # assert False

        # get rid of floes that intersect the dilated land mask
        condition = np.isin(watershed, np.unique(watershed[lmd & (watershed > 1)]))
        mask_image(watershed, condition, 1)

        # Test refactoring
        im4[np.isin(im4, np.unique(im4[(lmd == True) & (im4 > 1)]))] = 1
        assert np.array_equal(watershed, im4)
        # assert False

        # set the open water and already identified floes to no
        # watershed[~ice_mask] = 1
        mask_image(watershed, ~ice_mask, 1)

        # Test refactoring
        im4[ice_mask == False] = 1
        assert np.array_equal(watershed, im4)
        # assert False

        # get rid of ones that are too small
        area_lim = (it) ** 4
        props = skimage.measure.regionprops_table(
            watershed, properties=["label", "area"]
        )
        df = pd.DataFrame.from_dict(props)

        # Test refactoring
        _props = skimage.measure.regionprops_table(im4, properties=["label", "area"])
        _df = pd.DataFrame.from_dict(_props)
        assert df.equals(_df)
        # assert False

        condition = np.isin(watershed, df[df.area < area_lim].label.values)
        mask_image(watershed, condition, 1)
        # watershed[np.isin(watershed, df[df.area < area_lim].label.values)] = 1

        # Test refactoring
        im4[np.isin(im4, _df[_df.area < area_lim].label.values)] = 1
        assert np.array_equal(watershed, im4)
        # assert False


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

        # watershed[watershed < 2] = 0
        mask_image(watershed, watershed < 2, 0)
        output += watershed

        # Test refactoring
        im4[im4 < 2] = 0
        _output = im4 + _output
        assert np.array_equal(output, _output)
        # assert False


    output = opening(output)

    return output, red_c
