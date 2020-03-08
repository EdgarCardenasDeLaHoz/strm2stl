"""
Main module
"""
from itertools import product

import numpy as np
from scipy.ndimage import rotate
from PIL import Image

import os 
import numpy as np
import pandas as pd
import math

# Image and Visualization Tools
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon,MultiPolygon
from descartes import PolygonPatch


def get_section(root: str, ul: np.ndarray, lr: np.ndarray) -> np.ndarray:
    """
    Given root dir return rasterized mosaic DEM from rectangle bounded by grid
    coordinates given by upper left coordinates `ul` and lower right
    coordinates `lr` in the downampled image space.
    """
    TILE_SHAPE = 6000, 6000
    
    downshape = np.array(lr) - np.array(ul)
    outshape = downshape * np.array(TILE_SHAPE)

    out = np.zeros(outshape, "i2")
    for x, y in product(range(ul[0], lr[0]), range(ul[1], lr[1])):
        pathname = os.path.join(root, f"srtm_{x:02d}_{y:02d}.tif")
        
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

## Utility functions
def get_dem_geo(Root,North,South,East,West):
    "Uses convention where Y is Long and X is Lat"
    #X1,X2,Y1,Y2 = North,South,East,West
    TileX1,TileY1,X1,Y1 = geo_2_tile_pixel(North,West)
    TileX2,TileY2,X2,Y2 = geo_2_tile_pixel(South,East)
    X2 = (TileX2 - TileX1)*6000 + X2
    Y2 = (TileY2 - TileY1)*6000 + Y2

    DEM = get_section(Root, (TileX1, TileY1), (TileX2+1,TileY2+1))               
    DEM = crop_region(DEM, (Y1, X1), (Y2, X2), 0)
    DEM = np.maximum(DEM,0)
    DEM = DEM - (np.min(DEM)*.9)
    DEM = DEM / 90.0 * 10
    return DEM

def map_coor_2_pixel(GeoLoc,GeoStart,round_val=False):
    
    tX1,tY1,pX1,pY1 = geo_2_tile_pixel( GeoStart[0],GeoStart[1],round_val )
    tX2,tY2,pX2,pY2 = geo_2_tile_pixel( GeoLoc[0],GeoLoc[1],round_val )
    
    px = (tX2 - tX1)*6000 + (pX2 - pX1)
    py = (tY2 - tY1)*6000 + (pY2 - pY1)
    
    return (px,py)

def geo_2_tile_pixel(Lat=0,Long=0,round_val=True):

    "Produces convention where X is Long and Y is Lat"
    tilX = np.int16(np.floor(Long / 5) + 36 + 1)
    tilY = np.int16(np.floor(-Lat / 5) + 12 + 1)
    
    pixX = (Long /5 - np.floor(Long / 5))*6000
    pixY = (-Lat /5 - np.floor(-Lat / 5))*6000
    
    if round_val:
        pixX = np.int64(np.round(pixX))
        pixY = np.int64(np.round(pixY))
    
    return (tilX,tilY,pixX,pixY)

def tile_num_2_geo_coor(TX=0,TY=0,PX=0,PY=0):

    Long = 180 * (TX-37)/36
    Long += np.round(PX / 6000 * 5,4)        
    Lat = 60 * (TY-13)/12
    Lat += np.round(PY / 6000 * 5,4)
    
    return (-Lat,Long)

## Taking Coor DB as input 
def GetDem(Root,Coor,idx): 
    
    Row = Coor.iloc[idx]
    DEM = get_section(Root, (Row["GridX1"], Row["GridY1"]), (Row["GridX2"]+1,Row["GridY2"]+1))               
    DEM = crop_region(DEM, (Row["X1"], Row["Y1"]), (Row["X2"],Row["Y2"]) ,0)
    DEM = np.maximum(DEM,0)
    DEM = DEM - (np.min(DEM)*.9)
    DEM = DEM / 90.0 * 10
    return DEM

def get_bounds_geo(Coor,idx):
    
    #Get North, West 
    X = list(Coor.loc[idx,["GridX1","GridY1","Y1","X1"]])
    GeoStart = tile_num_2_geo_coor(*X)
    #Get South, East 
    X = list(Coor.loc[idx,["GridX1","GridY1","Y2","X2"]])
    GeoEnd = tile_num_2_geo_coor(*X)
    bounds = np.array( [GeoStart[0],GeoEnd[0],GeoEnd[1],GeoStart[1]])
    
    #bounds is north, south, east, west
    return bounds

## Taking DEM as Input 
def export_dem_2_stl(DEM,OutDir,name):
    fn = os.path.join(OutDir, name + ".stl")
    numpy2stl.numpy2stl(DEM, fn, solid= True, mask_val=0 )

def draw_dem(DEM,bounds=None):
    
    fig=plt.figure(figsize=(10, 10))
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax.grid(True) 
    if bounds is not None:
        ax.set_xlim(bounds[3],bounds[2])
        ax.set_ylim(bounds[1],bounds[0])
        bounds = bounds[[3,2,1,0]]
    #if bounds    
    ax.imshow(DEM,cmap = "jet",aspect= 'equal',extent = bounds)
    
def resize_dem(DEM,outsize=0):
    res = 1.0    
    if outsize > 0:
        res = np.array(outsize / np.amax(DEM.shape))
        DEM = DEM*res
        DEM = resize(DEM, np.round(np.array(DEM.shape) * res))
    
    return DEM, res

def embed_lines(DEM, pts_pix, res=1):     
    
    im_lines = np.zeros(DEM.shape, dtype=np.uint8)
    for x,y in pts_pix:
        x = x*res; y = y*res
        for i in range(len(x)-1):
            rr, cc, val = line_aa(int(round(y[i])), int(round(x[i])), int(round(y[i+1])), int(round(x[i+1])))
            im_lines[rr, cc] = val*255
                
    im_lines = morphology.binary_dilation(im_lines, morphology.disk(2))         
    DEM[im_lines>0] = DEM[im_lines>0] - (.15 * np.max(DEM)-np.min(DEM))
    
    return DEM
