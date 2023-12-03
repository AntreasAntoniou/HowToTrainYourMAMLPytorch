# Import necessary libraries and components for test.
import numpy as np
import pytest
from PIL import Image

from data import ImageRotator, RotationConfig


@pytest.mark.parametrize(
    "degrees,output_channels", [(90, 1), (180, 3), (45, 1)]
)
def test_image_rotator(degrees, output_channels):
    # Create a RotationConfig and ImageRotator instances
    config = RotationConfig(degrees=degrees, output_channels=output_channels)
    rotator = ImageRotator(config=config)

    # Generate a test image - a white square
    test_image = np.ones((10, 10, 3), dtype=np.uint8) * 255

    # Apply rotation
    rotated_image = rotator(test_image)

    # Check if image has been correctly rotated and channel adjusted
    assert (
        rotated_image.shape[:2] == test_image.shape[:2]
    ), "Rotated image dimensions should match original"
    if output_channels == 3:
        assert (
            rotated_image.ndim == 3 and rotated_image.shape[2] == 3
        ), "Rotated image should have 3 channels if output_channels is set to 3"
    elif output_channels == 1:
        assert rotated_image.ndim == 2 or (
            rotated_image.ndim == 3 and rotated_image.shape[2] == 1
        ), "Rotated image should have 1 channel if output_channels is set to 1"

    # Optionally, for arbitrary angles, you can compare with a PIL rotation
    if degrees % 90 != 0:
        pil_rotated_image = Image.fromarray(test_image).rotate(
            degrees, resample=Image.BICUBIC
        )
        pil_rotated_image = np.array(pil_rotated_image)
        assert (
            pil_rotated_image.shape == rotated_image.shape
        ), "Shapes from PIL and custom rotation do not match"
