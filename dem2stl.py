"""
Main module
"""

import os.path
from itertools import product

import numpy as np
from scipy.ndimage import rotate
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
    for x, y in product(range(ul[0], lr[0]), range(ul[1], lr[1])):
        pathname = os.path.join(root, f"srtm_{x:02d}_{y:02d}.tif")
        
        print(pathname)


        try:
            out[(x - ul[0]) * TILE_SHAPE[0]:(x - ul[0]+1) * TILE_SHAPE[0],
                (y - ul[1]) * TILE_SHAPE[1]:(y - ul[1]+1) * TILE_SHAPE[1]
            ] = np.array(Image.open(pathname))[:TILE_SHAPE[0], :TILE_SHAPE[1]].T
        except FileNotFoundError:
            print(f"srtm_{x:02d}_{y:02d}.tif file not found")
            continue

    return out.T


def crop_region(DEM,ul,lr,theta):
    """
    Given rasterized mosaic DEM and rectangle bounded by grid
    coordinates given by upper left coordinates `ul` and lower right
    coordinates `lr` crop and rotate image 
    """
    if not theta==0:
        DEM = rotate(DEM, theta, axes=(1, 0), reshape=True)
    DEM = DEM[ul[0]:lr[0], ul[1]:lr[1]]
    
    
    return DEM 
