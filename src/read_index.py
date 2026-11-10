"""
read_index.py — reads the Flona window from Landsat-C2-L2 reflectance bands
(21.12.22 LC09; 29.12.22 LC08), converts DN -> reflectance, computes a
set of spectral indices, and exports a CSV (one row per Flona pixel).

 Outputs:
   dados/pixels.csv           - full feature table (features + raw bands)
   figuras/indices_histograma.png - distribution plots of the main indices
"""

import os
import sys
import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- project layout (everything inside Trabalho_flona) --------------------
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEWPROJ = os.path.join(PROJECT, "recuperacao_queimadas")   # root of the new project
SRC     = os.path.join(NEWPROJ, "src")
DAD     = os.path.join(NEWPROJ, "dados")
FIGR    = os.path.join(NEWPROJ, "figuras")
os.makedirs(DAD, exist_ok=True)
os.makedirs(FIGR, exist_ok=True)

# ---- data paths (TIFs live inside the project, not in Trabalho_flona root) --
DIR_21 = os.path.join(NEWPROJ, "dados", "21.12.22")
DIR_29 = os.path.join(NEWPROJ, "dados", "29.12.22")
BAND_FILES = {
    "21": {
        "B4": os.path.join(DIR_21, "LC09_L2SP_221071_20221221_20230317_02_T2_SR_B4.TIF"),
        "B5": os.path.join(DIR_21, "LC09_L2SP_221071_20221221_20230317_02_T2_SR_B5.TIF"),
        "B6": os.path.join(DIR_21, "LC09_L2SP_221071_20221221_20230317_02_T2_SR_B6.TIF"),
        "B7": os.path.join(DIR_21, "LC09_L2SP_221071_20221221_20230317_02_T2_SR_B7.TIF"),
        "B2": os.path.join(DIR_21, "LC09_L2SP_221071_20221221_20230317_02_T2_SR_B2.TIF"),
    },
    "29": {
        "B4": os.path.join(DIR_29, "LC08_L2SP_221071_20221229_20230104_02_T1_SR_B4.TIF"),
        "B5": os.path.join(DIR_29, "LC08_L2SP_221071_20221229_20230104_02_T1_SR_B5.TIF"),
        "B6": os.path.join(DIR_29, "LC08_L2SP_221071_20221229_20230104_02_T1_SR_B6.TIF"),
        "B7": os.path.join(DIR_29, "LC08_L2SP_221071_20221229_20230104_02_T1_SR_B7.TIF"),
        "B2": os.path.join(DIR_29, "LC08_L2SP_221071_20221229_20230104_02_T1_SR_B2.TIF"),
    },
}

for d in ("21", "29"):
    for b in BAND_FILES[d]:
        assert os.path.exists(BAND_FILES[d][b]), f"missing {BAND_FILES[d][b]}"

SR = 0.0000275   # Landsat C2 L2 SR scale
SO = -0.2        # Landsat C2 L2 SR offset

# ---- Flona window (recomputed from the polygon + actual GT) ----------------
SRC_B5 = os.path.join(DIR_21, "LC09_L2SP_221071_20221221_20230317_02_T2_SR_B5.TIF")
_ds = gdal.Open(SRC_B5)
gt = _ds.GetGeoTransform()
ox, oy, px, py = gt[0], gt[3], gt[1], gt[5]
_ds = None

_shp = "/home/rafael/Documentos/psot/Trabalho_flona/inicial/FlonaBSB.shp"
from shapely.geometry import Polygon as SPoly, Point
from shapely.ops import unary_union
import shapefile

r = shapefile.Reader(_shp)
polys = [SPoly(s.points) for s in r.shapes()]
poly_geo = unary_union(polys)

# UTM polygon (for point-in-polygon)
_TO_utm = __import__("pyproj").Transformer.from_crs("EPSG:4326", "EPSG:32623", always_xy=True)
pts_utm = [_TO_utm.transform(x, y) for x, y in poly_geo.exterior.coords]
poly_utm = SPoly(pts_utm)

def _to_window(r0, c0, r1, c1):
    return (r0, c0, r1 - r0 + 1, c1 - c0 + 1)

LEAF_R0, LEAF_C0, LEAF_R1, LEAF_C1 = 3221, 3136, 3439, 3364
WIN_R = LEAF_R1 - LEAF_R0 + 1
WIN_C = LEAF_C1 - LEAF_C0 + 1
print(f"Flona window: {WIN_R} rows x {WIN_C} cols")

def read_band(path, r0, c0, r1, c1):
    ds = gdal.Open(path)
    b = ds.GetRasterBand(1)
    arr = b.ReadAsArray(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
    ds = None
    return arr

def dn_to_refl(dn):
    """DN -> surface reflectance (C2 L2), NaN where DN == 0 (fill)"""
    r = dn.astype(np.float32) * SR + SO
    r[dn == 0] = np.nan
    return r

# ---- load reflectance of the window --------------------------------------
print("\nReading bands ...")
refl = {}
for date in ("21", "29"):
    refl[date] = {b: dn_to_refl(read_band(BAND_FILES[date][b], LEAF_R0, LEAF_C0, LEAF_R1, LEAF_C1))
                  for b in ("B4", "B5", "B6", "B7", "B2")}
    print(f"  {date}: { {b: refl[date][b].shape for b in refl[date]} }")

# ---- index formulas ------------------------------------------------------
def idx_nbr(rr, r1):
    return (rr - r1) / (rr + r1 + 1e-9)

def idx_bais2(rr, sw2):
    return (rr - sw2) / (rr + sw2 + 1e-9)

def idx_ndvi(nr, rd):
    return (nr - rd) / (nr + rd + 1e-9)

def idx_evi(nr, rd, bd, g=2.5, c1=6.0, c2=7.5, L=1.0):
    return g * (nr - rd) / (nr + c1 * rd - c2 * bd + L)

def idx_savi(nr, rd, L=0.5):
    return (1 + L) * (nr - rd) / (nr + rd + L)

def idx_frp(sw2, nr):
    return sw2 / (nr + 1.0)

def idx_bai(sw1, nr):
    return (sw1 - nr) / (sw1 + nr + 1e-9)

def idx_nbr2(nr, sw2):
    return (nr ** 2 - sw2 ** 2) / (nr ** 2 + sw2 ** 2 + 1e-9)

print("Computing indices ...")
I21 = {
    "NBR_21": idx_nbr(refl["21"]["B5"], refl["21"]["B6"]),
    "NBR_29": idx_nbr(refl["29"]["B5"], refl["29"]["B6"]),
    "dNBR": idx_nbr(refl["21"]["B5"], refl["21"]["B6"]) -
            idx_nbr(refl["29"]["B5"], refl["29"]["B6"]),
    "BAIS2": idx_bais2(refl["21"]["B5"], refl["21"]["B7"]),
    "BAIS2_29": idx_bais2(refl["29"]["B5"], refl["29"]["B7"]),
    "NDVI_21": idx_ndvi(refl["21"]["B5"], refl["21"]["B4"]),
    "NDVI_29": idx_ndvi(refl["29"]["B5"], refl["29"]["B4"]),
    "EVI_21": idx_evi(refl["21"]["B5"], refl["21"]["B4"], refl["21"]["B2"]),
    "EVI_29": idx_evi(refl["29"]["B5"], refl["29"]["B4"], refl["29"]["B2"]),
    "SAVI_21": idx_savi(refl["21"]["B5"], refl["21"]["B4"]),
    "SAVI_29": idx_savi(refl["29"]["B5"], refl["29"]["B4"]),
    "FRP_21": idx_frp(refl["21"]["B7"], refl["21"]["B5"]),
    "FRP_29": idx_frp(refl["29"]["B7"], refl["29"]["B5"]),
    "BAI_21": idx_bai(refl["21"]["B6"], refl["21"]["B5"]),
    "BAI_29": idx_bai(refl["29"]["B6"], refl["29"]["B5"]),
    "NBR2_21": idx_nbr2(refl["21"]["B5"], refl["21"]["B7"]),
    "NBR2_29": idx_nbr2(refl["29"]["B5"], refl["29"]["B7"]),
}

# ---- build DataFrame (one row per pixel) -----------------------------------
print("Building dataframe ...")
rows = []
for i in range(WIN_R):
    for j in range(WIN_C):
        d = {
            "row": LEAF_R0 + i, "col": LEAF_C0 + j,
            "red21": refl["21"]["B4"][i, j], "nir21": refl["21"]["B5"][i, j],
            "swir1_21": refl["21"]["B6"][i, j], "swir2_21": refl["21"]["B7"][i, j],
            "b2_21": refl["21"]["B2"][i, j],
            "red29": refl["29"]["B4"][i, j], "nir29": refl["29"]["B5"][i, j],
            "swir1_29": refl["29"]["B6"][i, j], "swir2_29": refl["29"]["B7"][i, j],
            "b2_29": refl["29"]["B2"][i, j],
        }
        d.update({k: v[i, j] for k, v in I21.items()})
        rows.append(d)

df = pd.DataFrame(rows)
CSV = os.path.join(DAD, "pixels.csv")
df.to_csv(CSV, index=False, float_format="%.7f")
print(f"\nSaved {len(df)} pixels -> {CSV}")

# ---- statistics -----------------------------------------------------------
print("\nStatistics (Flona window):")
print(df.describe().round(4).to_string())

# ---- histograms -----------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
cols_hist = ["NBR_21", "NBR_29", "dNBR", "BAIS2", "EVI_21", "SAVI_21", "FRP_21", "BAI_21", "NBR2_21"]
for ax, col in zip(axes.flat, cols_hist):
    ax.hist(df[col].dropna(), bins=40, color="steelblue", alpha=0.75)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Freq")
fig.suptitle("Flona window – index distributions (21.12.22 vs 29.12.22)")
fig.tight_layout()
fig.savefig(os.path.join(FIGR, "indices_histograma.png"), dpi=150)
plt.close(fig)
print("Saved -> figuras/indices_histograma.png")
