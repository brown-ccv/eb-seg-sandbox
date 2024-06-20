import numpy as np
from ebfloeseg.masking import mask_image


def test_mask_image():
    img = np.ones((5, 5))
    mask = np.array(
        (
            [
                [True, False, True, False, False],
                [False, False, True, True, False],
                [False, True, False, True, True],
                [False, False, True, True, False],
                [False, True, True, True, False],
            ]
        )
    )
    mask_image(img, mask)
    arr = np.array(
        (
            [
                [0.0, 1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    assert np.allclose(img, arr)
