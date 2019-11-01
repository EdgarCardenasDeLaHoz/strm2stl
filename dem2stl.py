"""
Main module
"""

import os.path
from itertools import product

import numpy as np
from PIL import Image


TILE_SHAPE = 6000, 6000
def get_section(root: str, ul: np.ndarray, lr: np.ndarray) -> np.ndarray:
    """
    Given root dir return rasterized mosaic DEM from rectangle bounded by grid
    coordinates given by upper left coordinates `ul` and lower right
    coordinates `lr` in the downampled image space.
    """

    downshape = np.array(lr) - np.array(ul)
    outshape = downshape * np.array(TILE_SHAPE)
    out = np.zeros(outshape, "i2")
    for x, y in product(range(*ul), range(*lr)):
        pathname = os.path.join(root, f"strm_{x:02d}_{y:02d}.tif")
        try:
            out[(x - lr[0]) * TILE_SHAPE[0]:(x - lr[0]+1) * TILE_SHAPE[0],
                (y - lr[1]) * TILE_SHAPE[1]:(y - lr[1]+1) * TILE_SHAPE[1]
            ] = np.array(Image.open(pathname))[:TILE_SHAPE[0], :TILE_SHAPE[1]]
        except FileNotFoundError:
            continue

    return out.T
