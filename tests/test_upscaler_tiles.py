import numpy as np
from PIL import Image

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


def test_normalize_output_caps_size_and_restores_binary_alpha():
    rgba = np.zeros((400, 800, 4), dtype=np.uint8)
    rgba[..., :3] = (120, 30, 220)
    rgba[:, :400, 3] = 255
    source = Image.fromarray(rgba)

    result = Upscaler._normalize_output(source, (300, 300), "binary")

    assert result.size == (300, 150)
    assert set(np.unique(np.asarray(result.getchannel("A")))) <= {0, 255}


def test_normalize_output_never_enlarges_result():
    source = Image.new("RGBA", (120, 80), (1, 2, 3, 255))

    result = Upscaler._normalize_output(source, (600, 600), "binary")

    assert result is source
