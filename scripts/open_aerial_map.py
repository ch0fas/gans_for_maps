import os
import requests
import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image

# =============================
# Config
# =============================

OUTPUT_DIR = "./data/raw/oam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_TILES_PER_SCENE = 150   # Max tiles per scene
TILE_SIZE    = 256           # 256px per training tile
TILE_STRIDE  = 128           # Shifts tile every 128px, allowing overlap
MAX_SCENE_PX = 4096          # Max of 4096px per scene
MAX_GSD_METERS = 50          # Max Ground Sample Distance (GSD)
MIN_GSD_METERS = 0.1         # Min Ground Sample Distance (GSD)
RESULTS_PER_CITY = 20

OAM_API = "http://api.openaerialmap.org/meta"


# Only cities with confirmed OAM coverage
# [lon_min, lat_min, lon_max, lat_max]

CITIES = {
    "mexico_city":    [-99.20, 19.38, -99.08, 19.48],    
    "merida":         [-89.85, 20.75, -89.35, 21.20],    
    "cancun":         [-86.90, 21.11, -86.78, 21.21],    
    "santiago":       [-70.71, -33.51, -70.59, -33.39],  
    "buenos_aires":   [-58.65, -34.85, -58.15, -34.35],  
    "sao_paulo":      [-46.90, -23.80, -46.40, -23.30],  
    "rio_de_janeiro": [-43.45, -23.10, -42.95, -22.65],  
    "quito":          [-78.54,  -0.25, -78.42,  -0.15],  
    "kathmandu":      [ 85.28,  27.65,  85.40,  27.75],  
}



# =============================
# OAM search
# =============================

def search_oam(bbox: list, limit: int = RESULTS_PER_CITY) -> list:
    """Query OAM catalog for imagery within bbox, filtered by GSD, sharpest first."""
    lon_min, lat_min, lon_max, lat_max = bbox
    r = requests.get(OAM_API, params={
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "limit": limit,
    }, timeout=30)
    r.raise_for_status()

    results = r.json().get("results", [])
    filtered = [
        item for item in results
        if item.get("gsd") is not None
        and MIN_GSD_METERS <= item["gsd"] <= MAX_GSD_METERS
    ]
    filtered.sort(key=lambda x: x["gsd"])  # sharpest first
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
    h, w = ds.height, ds.width
    if max_px is None or max(h, w) <= max_px:
        return h, w
    scale = max_px / max(h, w)
    return int(h * scale), int(w * scale)

def download_scene(url, out_h, out_w):
    """
    Open GeoTIFF over HTTPS at (out_h, out_w).
    Returns uint8 RGB (H, W, 3) or raises ValueError if not valid RGB.
    """
    with rasterio.open(url) as ds:
        data = ds.read(
            out_shape=(ds.count, out_h, out_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

    n_bands = data.shape[0]

    if n_bands == 1:
        raise ValueError("Grayscale — not suitable for RGB GAN training")
    if n_bands < 3:
        raise ValueError(f"Only {n_bands} band(s) — need at least 3")

    r = clip_to_uint8(data[0])
    g = clip_to_uint8(data[1])
    b = clip_to_uint8(data[2])

    # Reject false-RGB (all bands identical — e.g. Bogota-style grayscale stored as 3-band)
    if np.mean(np.abs(r.astype(int) - g.astype(int))) < 2 and \
       np.mean(np.abs(r.astype(int) - b.astype(int))) < 2:
        raise ValueError("Bands near-identical (false RGB) — skipping")

    return np.stack([r, g, b], axis=-1)

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
        print(f"  No scenes found")
        continue

    print(f"  Found {len(results)} scene(s)")

    for item in results:
        title = item.get("title", item.get("_id", "unknown"))
        gsd   = item.get("gsd", "?")
        url   = item.get("uuid")

        if not url:
            print(f"  [SKIP] No download URL for {title}")
            continue

        print(f"  Downloading: {title}  |  GSD: {gsd}m")

        try:
            with rasterio.open(url) as ds:
                out_h, out_w = compute_out_shape(ds, MAX_SCENE_PX)
                print(f"    Native: {ds.height}×{ds.width}px → reading at {out_h}×{out_w}px")
            rgb = download_scene(url, out_h, out_w)
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        n_saved = 0
        for row, col, tile in tile_array(rgb, TILE_SIZE, TILE_STRIDE):
            if n_saved >= MAX_TILES_PER_SCENE:
                break
            safe_title = title.replace("/", "_").replace(" ", "_")[:60]
            fname = f"{city_name}_{safe_title}_r{row:04d}_c{col:04d}.png"
            Image.fromarray(tile, mode="RGB").save(os.path.join(OUTPUT_DIR, fname))
            n_saved += 1

        total_tiles += n_saved
        print(f"    → {n_saved} tiles saved")

print(f"\nAll done. {total_tiles} total tiles in {OUTPUT_DIR}/")