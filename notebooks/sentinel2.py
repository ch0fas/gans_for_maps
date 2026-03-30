import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image
from pystac_client import Client
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================
# Config
# =============================

OUTPUT_DIR = "./data/raw/sentinel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENE_SIZE  = 1024                              # Downloads 1024px satellite images 
TILE_SIZE   = 128                               # Extracts 128px tiles from scene
TILE_STRIDE = 64                                # Shifts tile every 64px, allowing overlap

MAX_CLOUD_COVER     = 5                         # Cloud % - Searches for clear images
MAX_NODATA_FRACTION = 0.02                      # Prevents black areas
DATE_RANGE          = "2020-01-01/2023-12-31"   # Takes into account 3 years of images
MAX_SCENES_PER_CITY = 5                         # Extracts 5 scenes per city
BANDS               = ["red", "green", "blue"]  # RGB channels


# List of cities centered at their capitals
# Extracts a 500m radius from each city
# [lon_min, lat_min, lon_max, lat_max]
CITIES = {
    "mexico_city":    [-99.138, 19.428, -99.118, 19.448],   # Centro Histórico
    "guadalajara":    [-103.354, 20.673, -103.334, 20.693], # Centro / Analco
    "monterrey":      [-100.321, 25.669, -100.301, 25.689], # Barrio Antiguo
    "puebla":         [-98.207,  19.041, -98.187,  19.061], # Centro
    "bogota":         [-74.078,  4.608,  -74.058,  4.628],  # La Candelaria
    "lima":           [-77.033, -12.051, -77.013, -12.031], # Cercado de Lima
    "santiago":       [-70.654, -33.455, -70.634, -33.435], # Santiago Centro
    "buenos_aires":   [-58.386, -34.608, -58.366, -34.588], # San Telmo / Microcentro
    "sao_paulo":      [-46.638, -23.551, -46.618, -23.531], # Sé / República
    "rio_de_janeiro": [-43.175, -22.904, -43.155, -22.884], # Centro / Lapa
}

# =============================
# STAC client
# =============================

catalog = Client.open("https://earth-search.aws.element84.com/v1")

# =============================
# Helpers
# =============================

def clip_to_uint8(arr, lo=2, hi=98):
    """Percentile clip float32 → uint8. Format conversion only, not GAN normalization."""
    p_lo = np.percentile(arr, lo)
    p_hi = np.percentile(arr, hi)
    clipped = np.clip(arr, p_lo, p_hi)
    return ((clipped - p_lo) / (p_hi - p_lo + 1e-6) * 255).astype(np.uint8)


def fetch_band(url: str, size: int, resampling=Resampling.bilinear) -> np.ndarray:
    """COG windowed read at target size via HTTP range requests."""
    with rasterio.open(url) as ds:
        return ds.read(
            1,
            out_shape=(size, size),
            resampling=resampling,
        ).astype(np.float32)


def check_nodata(scl_url: str, check_size: int = 128) -> float:
    """Quick SCL nodata check. SCL==0 → outside swath (black triangles)."""
    with rasterio.open(scl_url) as ds:
        scl = ds.read(1, out_shape=(check_size, check_size), resampling=Resampling.nearest)
    return np.sum(scl == 0) / scl.size


def tile_array(rgb: np.ndarray, tile_size: int, stride: int):
    """Yield (row, col, tile) skipping tiles with >10% black pixels."""
    h, w, _ = rgb.shape
    for row in range(0, h - tile_size + 1, stride):
        for col in range(0, w - tile_size + 1, stride):
            tile = rgb[row:row + tile_size, col:col + tile_size]
            black_frac = np.sum(np.all(tile == 0, axis=-1)) / (tile_size * tile_size)
            if black_frac > 0.10:
                continue
            yield row, col, tile


# =============================
# Scene download + tiling
# =============================

def download_and_tile(item, city_name: str) -> int:
    cloud = item.properties.get("eo:cloud_cover", "?")
    date  = item.datetime.date() if item.datetime else "unknown"

    scl_asset = item.assets.get("scl")
    if scl_asset is not None:
        nodata_frac = check_nodata(scl_asset.href)
        if nodata_frac > MAX_NODATA_FRACTION:
            print(f"  [REJECT] {item.id} | nodata={nodata_frac:.1%}")
            return 0
        print(f"  [OK]  {item.id} | {date} | cloud: {cloud:.1f}% | nodata: {nodata_frac:.1%}")
    else:
        print(f"  [OK]  {item.id} | {date} | cloud: {cloud:.1f}% | no SCL")

    urls = {}
    for band in BANDS:
        asset = item.assets.get(band)
        if asset is None:
            print(f"    [SKIP] '{band}' missing")
            return 0
        urls[band] = asset.href

    band_arrays = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch_band, urls[b], SCENE_SIZE): b for b in BANDS}
        for future in as_completed(futures):
            band = futures[future]
            try:
                band_arrays[band] = future.result()
            except Exception as e:
                print(f"    [ERROR] {band}: {e}")
                return 0

    if len(band_arrays) != 3:
        return 0

    rgb = np.stack([
        clip_to_uint8(band_arrays["red"]),
        clip_to_uint8(band_arrays["green"]),
        clip_to_uint8(band_arrays["blue"]),
    ], axis=-1)

    n_saved = 0
    for row, col, tile in tile_array(rgb, TILE_SIZE, TILE_STRIDE):
        fname = f"{city_name}_{item.id}_r{row:04d}_c{col:04d}.png"
        Image.fromarray(tile, mode="RGB").save(os.path.join(OUTPUT_DIR, fname))
        n_saved += 1

    print(f"    → {n_saved} tiles  (scene: {rgb.shape[0]}×{rgb.shape[1]}px, {TILE_SIZE}px tiles)")
    return n_saved


# =============================
# Main loop
# =============================

total_tiles = 0

for city_name, bbox in CITIES.items():
    print(f"\n{'='*55}")
    print(f"  {city_name.upper()}")
    print(f"{'='*55}")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
        sortby="-properties.eo:cloud_cover",
        max_items=MAX_SCENES_PER_CITY,
    )

    items = list(search.items())
    print(f"  Found {len(items)} candidate scenes")

    if not items:
        print("  No scenes — try widening DATE_RANGE or MAX_CLOUD_COVER")
        continue

    for item in items:
        total_tiles += download_and_tile(item, city_name)

print(f"\nAll done. {total_tiles} tiles saved to {OUTPUT_DIR}/")
print(f"Coverage per tile: ~{int(500 * TILE_SIZE / SCENE_SIZE)}×{int(500 * TILE_SIZE / SCENE_SIZE)}m  |  Stride: {TILE_STRIDE}px")