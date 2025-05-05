import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import rasterio
from rasterio.plot import show
import os

# Optional: turn off warnings
import warnings
warnings.filterwarnings("ignore")

# === Sample Data ===
# Replace this with your actual DataFrame (must contain 'Latitude', 'Longitude', 'Cluster')
data = {
    'Latitude': [45.4215, 49.2827, 43.6532, 45.5017, 53.5461],
    'Longitude': [-75.6972, -123.1207, -79.3832, -73.5673, -113.4938],
    'Cluster': [0, 1, 0, -1, 1]
}
df = pd.DataFrame(data)

# === Convert to GeoDataFrame and reproject ===
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)  # Web Mercator for plotting with raster

# === Plotting Function ===
def plot_clustered_locations(gdf, raster_path, title="Museums Clustered by Proximity"):
    fig, ax = plt.subplots(figsize=(15, 10))

    # Load and show the basemap using rasterio
    with rasterio.open(raster_path) as src:
        show(src, ax=ax)

    # Separate clustered and noise points
    clustered = gdf[gdf['Cluster'] != -1]
    noise = gdf[gdf['Cluster'] == -1]

    # Plot points
    clustered.plot(ax=ax, column='Cluster', cmap='tab10', markersize=50, edgecolor='black', alpha=0.7, legend=False)
    noise.plot(ax=ax, color='black', markersize=50, edgecolor='red', alpha=1, label='Noise')

    # Formatting
    plt.title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel("Longitude (Web Mercator)")
    plt.ylabel("Latitude (Web Mercator)")
    plt.tight_layout()
    plt.legend()
    plt.show()

# === Call the function with your local TIFF path ===
local_tif = "./Canada.tif"  # Make sure this path is correct
plot_clustered_locations(gdf, local_tif)

