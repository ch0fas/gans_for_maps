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

SCENE_SIZE  = 4096   # Downloads 4096px satellite images
TILE_SIZE   = 512    # Extracts 512px per scene
TILE_STRIDE = 256    # Shifts tile every 256px, allowing overlap

MAX_CLOUD_COVER     = 5      # Cloud % - Searches for clear images
MAX_NODATA_FRACTION = 0.05   # Prevents black areas
DATE_RANGE          = "2020-01-01/2023-12-31"   # Takes into account 3 years of images
MAX_SCENES_PER_AOI  = 2      # Extracts 2 scenes per location
BANDS               = ["red", "green", "blue"]  # RGB channels


# List of geographic locations (areas of interest)
# Chosen for visual diversity 
# [lon_min, lat_min, lon_max, lat_max]

AOIS = {
    # --- Coasts & Deltas ---
    "amazon_delta":         [ -51.5, -1.5,  -49.0,  1.0],   # Brazil — huge river mouth
    "mekong_delta":         [ 105.0,  9.5,  107.0, 11.0],   # Vietnam — intricate waterways
    "ganges_delta":         [  88.5, 21.5,   91.0, 23.5],   # Bangladesh/India — Sundarbans
    "patagonian_coast":     [ -67.0,-46.0,  -64.5,-44.0],   # Argentina — fjords & inlets

    # --- Mountain Ranges ---
    "andes_peru":           [ -77.0,-14.0,  -74.5,-11.5],   # Peru — rugged high Andes
    "alps_switzerland":     [   7.5, 46.0,   10.0, 47.5],   # Switzerland — dramatic peaks
    "caucasus":             [  43.0, 42.0,   46.0, 43.5],   # Georgia/Russia — high ridges
    "atlas_morocco":        [  -6.0, 30.5,   -3.0, 32.0],   # Morocco — Atlas range

    # --- Deserts ---
    "sahara_algeria":       [   2.0, 25.0,    5.0, 27.0],   # Algeria — pure erg dunes
    "gobi_mongolia":        [ 103.0, 42.0,  106.0, 44.0],   # Mongolia — vast steppe/desert
    "namib_desert":         [  15.0,-24.0,   17.5,-22.0],   # Namibia — coastal desert

    # --- Islands & Archipelagos ---
    "greek_islands":        [  25.0, 36.5,   27.5, 38.0],   # Greece — Aegean islands
    "indonesia_sulawesi":   [ 122.0, -4.0,  124.5, -2.0],   # Indonesia — complex coastline
    "scottish_highlands":   [  -5.5, 57.0,   -3.0, 58.5],   # Scotland — lochs & glens

    # --- Plains & Steppe ---
    "hungarian_plain":      [  18.0, 46.5,   20.5, 48.0],   # Hungary — flat pannonian
    "siberian_plain":       [  68.0, 57.0,   71.0, 59.0],   # Russia — taiga/river network
    "serengeti":            [  34.0, -3.5,   36.5, -1.5],   # Tanzania — savanna plains

    # --- Forests & Jungles ---
    "amazon_interior":      [ -63.0, -8.0,  -60.5, -6.0],   # Brazil — dense rainforest
    "congo_basin":          [  23.0,  0.0,   25.5,  2.0],   # DRC — equatorial forest
    "borneo_interior":      [ 114.0,  1.0,  116.5,  3.0],   # Malaysia — rainforest

    # --- Lakes & Inland Seas ---
    "lake_titicaca":        [ -70.0,-16.5,  -68.5,-15.0],   # Peru/Bolivia — high altitude lake
    "great_rift_lakes":     [  29.0, -3.0,   31.5, -1.0],   # Tanzania — rift valley lakes
    "aral_sea_remnant":     [  59.0, 44.0,   61.5, 46.0],   # Kazakhstan — shrinking sea

    # --- Latin America ---
    "mexico_yucatan":       [ -90.5, 19.5,  -88.0, 21.0],   # Yucatan coast & cenotes
    "colombia_coast":       [ -77.5,  7.5,  -75.0,  9.0],   # Colombian Caribbean
    "chile_lake_district":  [ -73.0,-40.5,  -70.5,-38.5],   # Chile — lakes & volcanoes
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


def fetch_band(url: str, size: int, resampling=Resampling.average) -> np.ndarray:
    """
    COG windowed read at target size via HTTP range requests.
    Uses average resampling — better for downscaling large scenes.
    """
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
    """
    Yield (row, col, tile) skipping tiles with >10% black pixels.
    Also skips near-uniform tiles (featureless sky/ocean) — not useful for GAN training.
    """
    h, w, _ = rgb.shape
    for row in range(0, h - tile_size + 1, stride):
        for col in range(0, w - tile_size + 1, stride):
            tile = rgb[row:row + tile_size, col:col + tile_size]

            # Skip nodata tiles
            black_frac = np.sum(np.all(tile == 0, axis=-1)) / (tile_size * tile_size)
            if black_frac > 0.10:
                continue

            # Skip near-uniform tiles (std < 8 across all channels → featureless)
            if tile.std() < 8:
                continue

            yield row, col, tile


# =============================
# Scene download + tiling
# =============================

def download_and_tile(item, aoi_name: str) -> int:
    cloud = item.properties.get("eo:cloud_cover", "?")
    date  = item.datetime.date() if item.datetime else "unknown"

    # SCL nodata check
    scl_asset = item.assets.get("scl")
    if scl_asset is not None:
        nodata_frac = check_nodata(scl_asset.href)
        if nodata_frac > MAX_NODATA_FRACTION:
            print(f"  [REJECT] {item.id} | nodata={nodata_frac:.1%}")
            return 0
        print(f"  [OK]  {item.id} | {date} | cloud: {cloud:.1f}% | nodata: {nodata_frac:.1%}")
    else:
        print(f"  [OK]  {item.id} | {date} | cloud: {cloud:.1f}%")

    # Collect band URLs
    urls = {}
    for band in BANDS:
        asset = item.assets.get(band)
        if asset is None:
            print(f"    [SKIP] '{band}' missing")
            return 0
        urls[band] = asset.href

    # Parallel band download
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
        fname = f"{aoi_name}_{item.id}_r{row:04d}_c{col:04d}.png"
        Image.fromarray(tile, mode="RGB").save(os.path.join(OUTPUT_DIR, fname))
        n_saved += 1

    print(f"    → {n_saved} tiles  ({rgb.shape[0]}×{rgb.shape[1]}px scene, {TILE_SIZE}px tiles)")
    return n_saved


# =============================
# Main loop
# =============================

total_tiles = 0

for aoi_name, bbox in AOIS.items():
    print(f"\n{'='*55}")
    print(f"  {aoi_name.upper()}")
    print(f"{'='*55}")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
        sortby="-properties.eo:cloud_cover",   # clearest first
        max_items=MAX_SCENES_PER_AOI,
    )

    items = list(search.items())
    print(f"  Found {len(items)} candidate scenes")

    if not items:
        print("  No scenes — try widening DATE_RANGE or MAX_CLOUD_COVER")
        continue

    for item in items:
        total_tiles += download_and_tile(item, aoi_name)

print(f"\nAll done. {total_tiles} tiles saved to {OUTPUT_DIR}/")
print(f"Tile size: {TILE_SIZE}px  |  Stride: {TILE_STRIDE}px  |  Scene: {SCENE_SIZE}px")