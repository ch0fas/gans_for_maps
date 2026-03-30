import os
import math
import requests
import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================
# Config
# =============================

OUTPUT_DIR = "./data/raw/oam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TILE_SIZE    = 256    # Tiles of 256px
TILE_STRIDE  = 128    # Shifts tile every 64px, allowing overlap

# Set a max of 4096px per scene
MAX_SCENE_PX = 4096   

# MAx Ground Sample Distance (GSD) in meters
MAX_GSD_METERS = 1.0

# Bounding boxes for OAM search — radius of 5km, since OAM takes closer pictures
# [lon_min, lat_min, lon_max, lat_max]
CITIES = {
    "mexico_city":    [-99.20, 19.38, -99.08, 19.48],
    "guadalajara":    [-103.41, 20.64, -103.29, 20.74],
    "monterrey":      [-100.37, 25.63, -100.25, 25.73],
    "puebla":         [-98.24,  18.99, -98.12,  19.09],
    "bogota":         [-74.14,  4.57,  -74.02,  4.67 ],
    "lima":           [-77.08, -12.12, -76.96, -12.00],
    "santiago":       [-70.71, -33.51, -70.59, -33.39],
    "buenos_aires":   [-58.45, -34.66, -58.33, -34.54],
    "sao_paulo":      [-46.70, -23.60, -46.58, -23.48],
    "rio_de_janeiro": [-43.24, -22.94, -43.12, -22.82],
}

OAM_API = "http://api.openaerialmap.org/meta"

# =============================
# OAM search
# =============================

def search_oam(bbox, max_gsd=MAX_GSD_METERS, limit=20):
    """
    Query OAM catalog for imagery within bbox.
    Returns list of result dicts sorted by GSD (sharpest first).
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    params = {
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "limit": limit,
    }
    r = requests.get(OAM_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    results = data.get("results", [])
    if not results:
        return []

    # Filter by GSD and sort sharpest first
    filtered = []
    for item in results:
        gsd = item.get("gsd")          # meters/pixel, may be None
        if gsd is not None and gsd > max_gsd:
            continue
        filtered.append(item)

    filtered.sort(key=lambda x: x.get("gsd") or 999)
    return filtered

# =============================
# Helpers
# =============================

def clip_to_uint8(arr, lo=2, hi=98):
    """Percentile clip → uint8. Format conversion only, not GAN normalization."""
    p_lo = np.percentile(arr, lo)
    p_hi = np.percentile(arr, hi)
    clipped = np.clip(arr, p_lo, p_hi)
    return ((clipped - p_lo) / (p_hi - p_lo + 1e-6) * 255).astype(np.uint8)


def compute_out_shape(ds, max_px):
    """Compute (height, width) capped at max_px on the longest side, preserving aspect ratio."""
    h, w = ds.height, ds.width
    if max_px is None or max(h, w) <= max_px:
        return h, w
    scale = max_px / max(h, w)
    return int(h * scale), int(w * scale)


def download_scene(url, out_h, out_w):
    """
    Open a GeoTIFF over HTTPS and read it at (out_h, out_w).
    Handles 1-band (grayscale), 3-band (RGB), and 4-band (RGBA) images.
    Returns uint8 RGB numpy array (H, W, 3).
    """
    with rasterio.open(url) as ds:
        out_shape = (ds.count, out_h, out_w)
        data = ds.read(
            out_shape=out_shape,
            resampling=Resampling.bilinear,
        ).astype(np.float32)  # (bands, H, W)

    n_bands = data.shape[0]

    if n_bands == 1:
        # Grayscale → replicate to RGB
        band = clip_to_uint8(data[0])
        rgb = np.stack([band, band, band], axis=-1)
    elif n_bands >= 3:
        # RGB or RGBA — take first 3 bands
        r = clip_to_uint8(data[0])
        g = clip_to_uint8(data[1])
        b = clip_to_uint8(data[2])
        rgb = np.stack([r, g, b], axis=-1)
    else:
        raise ValueError(f"Unexpected band count: {n_bands}")

    return rgb  # (H, W, 3) uint8


def tile_array(rgb, tile_size, stride):
    """Yield (row, col, tile), skipping tiles with >10% black pixels."""
    h, w, _ = rgb.shape
    for row in range(0, h - tile_size + 1, stride):
        for col in range(0, w - tile_size + 1, stride):
            tile = rgb[row:row + tile_size, col:col + tile_size]
            black_frac = np.sum(np.all(tile == 0, axis=-1)) / (tile_size * tile_size)
            if black_frac > 0.10:
                continue
            yield row, col, tile

# =============================
# Main loop
# =============================

total_tiles = 0

for city_name, bbox in CITIES.items():
    print(f"\n{'='*55}")
    print(f"  {city_name.upper()}")
    print(f"{'='*55}")

    results = search_oam(bbox)

    if not results:
        print(f"  No OAM imagery found (try widening bbox or MAX_GSD_METERS)")
        continue

    print(f"  Found {len(results)} image(s)")

    for item in results:
        title    = item.get("title", item.get("_id", "unknown"))
        gsd      = item.get("gsd", "?")
        url      = item.get("uuid")      # direct GeoTIFF download URL in OAM

        if not url:
            print(f"  [SKIP] No download URL for {title}")
            continue

        print(f"  Downloading: {title}  |  GSD: {gsd}m")

        try:
            with rasterio.open(url) as ds:
                out_h, out_w = compute_out_shape(ds, MAX_SCENE_PX)
                print(f"    Native: {ds.height}×{ds.width}px → reading at {out_h}×{out_w}px")

            rgb = download_scene(url, out_h, out_w)

        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        n_saved = 0
        for row, col, tile in tile_array(rgb, TILE_SIZE, TILE_STRIDE):
            safe_title = title.replace("/", "_").replace(" ", "_")[:60]
            fname = f"{city_name}_{safe_title}_r{row:04d}_c{col:04d}.png"
            Image.fromarray(tile, mode="RGB").save(os.path.join(OUTPUT_DIR, fname))
            n_saved += 1

        total_tiles += n_saved
        print(f"    → {n_saved} tiles saved")

print(f"\nAll done. {total_tiles} total tiles in {OUTPUT_DIR}/")