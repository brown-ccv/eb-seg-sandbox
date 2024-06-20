import numpy as np
import rasterio
from numpy.testing import assert_array_equal
from ebfloeseg.masking import mask_image, create_land_mask, create_cloud_mask, maskrgb


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


expected_mask = np.array(
    [[False, False, True], [False, True, False], [True, False, False]]
)


def test_create_land_mask():
    # Create a dummy land mask
    land_mask = np.array([[0, 0, 75], [0, 75, 0], [75, 0, 0]])

    # Mock the rasterio.open function
    class MockRasterioOpen:
        land_mask = np.array([[0, 0, 75], [0, 75, 0], [75, 0, 0]])

        def __init__(self, lmfile):
            self.land_mask = land_mask

        def read(self):
            return [self.land_mask]

    rasterio.open = MockRasterioOpen

    # Call the create_land_mask function
    result = create_land_mask("dummy_file")

    # Assert the result
    assert_array_equal(result, expected_mask)


def test_create_cloud_mask():

    # Mock the rasterio.open function
    class MockRasterioOpen:
        def __init__(self, lmfile):
            self.cloud_mask = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]])

        def read(self):
            return [self.cloud_mask]

    rasterio.open = MockRasterioOpen

    # Call the create_cloud_mask function
    result = create_cloud_mask("dummy_file")

    # Assert the result
    assert_array_equal(result, expected_mask)

def test_maskrgb():
    # Create a dummy RGB image
    rgb = np.full((3, 3, 3), 255, dtype=np.uint8)
    # Create a dummy mask
    mask = np.array(
        [[False, False, True], [False, True, False], [True, False, False]]
    )
    # Call the maskrgb function
    maskrgb(rgb, mask)
    # Define the expected result
    expected_result = np.where(np.stack([mask for _ in range(3)], axis=-1), 0, 255)

    # Assert the result
    assert_array_equal(rgb, expected_result)
    