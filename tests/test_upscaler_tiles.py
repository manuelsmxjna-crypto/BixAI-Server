import numpy as np

from app.upscaler import Upscaler


def test_blend_tile_fills_destination_without_holes():
    destination = np.zeros((12, 12, 3), dtype=np.uint8)
    first = np.full((12, 8, 3), 40, dtype=np.uint8)
    second = np.full((12, 8, 3), 200, dtype=np.uint8)

    Upscaler._blend_tile(destination, first, 0, 0, 4, False, False)
    Upscaler._blend_tile(destination, second, 4, 0, 4, True, False)

    assert np.all(destination[:, :4] == 40)
    assert np.all(destination[:, 8:] == 200)
    assert np.all(destination[:, 4:8] > 40)
    assert np.all(destination[:, 4:8] < 200)


def test_blend_tile_softens_horizontal_and_vertical_boundaries():
    destination = np.full((10, 10, 3), 20, dtype=np.uint8)
    tile = np.full((6, 6, 3), 220, dtype=np.uint8)

    Upscaler._blend_tile(destination, tile, 4, 4, 4, True, True)

    assert 20 < destination[4, 4, 0] < destination[7, 7, 0] < 220
    assert destination[9, 9, 0] == 220

