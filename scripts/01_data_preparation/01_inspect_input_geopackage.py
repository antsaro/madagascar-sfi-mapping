"""
Inspect the raw SFI soil-profile GeoPackage: list layers, CRS, shape,
columns, dtypes, and preview the attribute table. Run this first to
confirm the input file structure before extraction.
"""

import geopandas as gpd

# Path to your file
path = "/kaggle/input/datasets/antsasarobidyran/data-sfi-nirs/Data_SFIv2.gpkg"

# List layers in the geopackage (useful if it has multiple layers)
import fiona
layers = fiona.listlayers(path)
print("Layers found:", layers)

# Read the (first) layer
gdf = gpd.read_file(path, layer=layers[0])

# Basic info
print("\nCRS:", gdf.crs)
print("Shape (rows, cols):", gdf.shape)
print("\nColumn names:")
print(gdf.columns.tolist())

print("\nData types:")
print(gdf.dtypes)

# Full attribute table (without geometry, for readability)
attribute_table = gdf.drop(columns="geometry")
print("\nAttribute table preview:")
display(attribute_table.head(20))

# If you want to see all rows (careful if large)
# display(attribute_table)
