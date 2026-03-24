#! /usr/bin/env python3
# This code takes the network nodes and applies DEM heights to each node so that slope and more accurate distance calculations can be made. For example, if two nodes are at different elevations, the actual distance between them is longer than the straight line distance. This can affect shortest path calculations and other network analysis. Another deterrent is slope, which can make walking more difficult. High slope paths might be avoided by walkers. 

# imports
import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window
from pyproj import Transformer

# load in the network nodes
nodes = gpd.read_file("/home/shared/cake/sub-projects/student_movement_model/network_files/walk_nodes_clean.geojson")
nodes = nodes.to_crs(epsg=4326)

# load in DEM
dem = rasterio.open("/home/shared/cake/cake_db/dem/georgia_tech_high_resolution_dem/ned19_n34x00_w084x50_ga_fultonco_2006.img")

# define transformer
_to_dem_crs = Transformer.from_crs("EPSG:4326", dem.crs, always_xy=True)  # (lon, lat) -> DEM CRS

# make function to get height at a lat lon point
def get_height_at_point(lat, lon):
    # convert lat lon to DEM coordinate reference system
    x, y = _to_dem_crs.transform(lon, lat)

    # ensure the point falls within the raster bounds
    if not (dem.bounds.left <= x <= dem.bounds.right and dem.bounds.bottom <= y <= dem.bounds.top):
        raise ValueError("Latitude/longitude point falls outside the DEM extent")

    row, col = dem.index(x, y)

    if not (0 <= row < dem.height and 0 <= col < dem.width):
        raise ValueError("Latitude/longitude point falls outside the DEM bounds")

    window = Window(col, row, 1, 1)
    height_block = dem.read(1, window=window)

    if height_block.size == 0:
        raise ValueError("Failed to read DEM height for the provided point")

    return float(height_block[0, 0])

# setup loop to add a height column to each node row

for index, row in nodes.iterrows():
    nodes.loc[index, "height"] = get_height_at_point(row["geometry"].y, row["geometry"].x)

# save the nodes with the same filename
nodes.to_file("/home/shared/cake/sub-projects/student_movement_model/network_files/walk_nodes_clean.geojson", driver="GeoJSON")

# load in edges
edges = gpd.read_file("/home/shared/cake/sub-projects/student_movement_model/network_files/walk_edges_clean.geojson")
edges = edges.to_crs(epsg=4326)

# each edge has start and end node. For each edge (row), get the two heights and calculate the difference in height and slope. Note that the heights are in meters but the edge length is in feet.

for index, row in edges.iterrows():
    start_height = nodes.loc[nodes["node_id"] == row["from_id"], "height"].values[0] * 3.28084 # convert to feet
    end_height = nodes.loc[nodes["node_id"] == row["to_id"], "height"].values[0] * 3.28084 # convert to feet
    height_diff = end_height - start_height
    total_length = np.sqrt(height_diff**2 + row["length_ft"]**2)
    slope = height_diff / row["length_ft"]
    edges.loc[index, "height_diff"] = height_diff
    edges.loc[index, "total_length"] = total_length
    edges.loc[index, "slope"] = slope

# save edges with the same filename
edges.to_file("/home/shared/cake/sub-projects/student_movement_model/network_files/walk_edges_clean.geojson", driver="GeoJSON")