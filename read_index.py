"""
read_index.py — reads the Flona window from Landsat-C2-L2 reflectance bands
(pre/post fire), converts DN -> reflectance, computes spectral indices,
and exports a CSV (one row per Flona pixel).

Descoberta automática:
  - Cenas:    toda subpasta de dados/ que contenha *_SR_B*.TIF
  - Data:     extraída do nome Landsat (_YYYYMMDD_)
              menor data = "pre" (antiga 21.12.22), maior = "post" (29.12.22)
  - Bandas:   casadas por sufixo _SR_B{2,4,5,6,7}.TIF
  - Janela:   derivada de inicial/FlonaBSB.shp + geotransform do B5 da cena "pre"

As colunas do CSV mantêm os sufixos "_21"/"_29" (herança do projeto antigo)
para não quebrar o classifier.py — "21" = pre, "29" = post.

 Outputs:
   dados/pixels.csv                - full feature table
   figuras/indices_histograma.png  - distribution plots
"""

import os
import sys
import glob
import re
import numpy 
import pandas
from osgeo import gdal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime

# - project layout (derived from __file__, nothing user-specific) 
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEWPROJ = os.path.join(PROJECT, "recuperacao_queimadas")
SRC     = os.path.join(NEWPROJ, "src")
DAD     = os.path.join(NEWPROJ, "dados")
FIGR    = os.path.join(NEWPROJ, "figuras")
os.makedirs(DAD, exist_ok=True)
os.makedirs(FIGR, exist_ok=True)

# make src/ importable to reuse utils.polygon_pixel_window
sys.path.insert(0, SRC)

SR = 0.0000275
SO = -0.2

#discover scenes / dates / bands 
_FOLDER_DATE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{2,4})$")
_BAND_RE        = re.compile(r"_SR_B(\d+)\.TIF$")
REQUIRED_BANDS  = ["B2", "B4", "B5", "B6", "B7"]

def _parse_folder_date(name):
    """'21.12.22' ou '21.12.2022' -> datetime.date. None se não casar."""
    m = _FOLDER_DATE_RE.match(name.strip())
    if not m:
        return None
    d, mo, y = (int(g) for g in m.groups())
    if y < 100:
        y = 2000 + y
    try:
        return datetime.date(y, mo, d)
    except ValueError:
        return None

def discover_scenes(dados_dir):
    """Return {datetime.date: {band: path}} para cada subpasta de dados/
    cujo nome seja dd.mm.yy (ou dd.mm.yyyy) e contenha *_SR_B*.TIF."""
    scenes = {}
    for entry in sorted(os.listdir(dados_dir)):
        folder = os.path.join(dados_dir, entry)
        if not os.path.isdir(folder):
            continue
        date = _parse_folder_date(entry)
        if date is None:
            print(f"  [skip] {entry}: nome de pasta não segue dd.mm.yy")
            continue
        tifs = glob.glob(os.path.join(folder, "*_SR_B*.TIF"))
        if not tifs:
            print(f"  [skip] {entry}: sem *_SR_B*.TIF")
            continue
        bands = {}
        for t in tifs:
            bm = _BAND_RE.search(t)
            if bm:
                bands[f"B{bm.group(1)}"] = t
        scenes[date] = bands
    return scenes

SCENES = discover_scenes(DAD)
if len(SCENES) < 2:
    raise RuntimeError(f"Preciso de >=2 pastas de data em {DAD}, achei: {sorted(SCENES)}")
if len(SCENES) > 2:
    print(f"AVISO: {len(SCENES)} cenas encontradas — usando a mais antiga e a mais nova")

dates_sorted = sorted(SCENES)
DATE_EARLY, DATE_LATE = dates_sorted[0], dates_sorted[-1]

if DATE_EARLY == DATE_LATE:
    raise RuntimeError("Datas early e late coincidem — confira as pastas")

print(f"Cenas: early={DATE_EARLY.isoformat()}  late={DATE_LATE.isoformat()}")

for d in (DATE_EARLY, DATE_LATE):
    missing = [b for b in REQUIRED_BANDS if b not in SCENES[d]]
    if missing:
        raise RuntimeError(f"Cena {d.isoformat()} sem bandas {missing}: tem {sorted(SCENES[d])}")

BAND_FILES = {DATE_EARLY: SCENES[DATE_EARLY], DATE_LATE: SCENES[DATE_LATE]}

#Flona window derived from the polygon (no hardcoded row/col) -
from utils import load_flona_polygon, polygon_pixel_window

shp_candidates = glob.glob(os.path.join(PROJECT, "inicial", "*.shp"))
if not shp_candidates:
    raise FileNotFoundError(f"Nenhum .shp em {os.path.join(PROJECT, 'inicial')}")
shp = shp_candidates[0]
poly_geo = load_flona_polygon(shp)

# pass src explicitly so utils doesn't fall back to its own hardcoded path
_ds_b5 = gdal.Open(BAND_FILES[DATE_PRE]["B5"])
LEAF_R0, LEAF_C0, LEAF_R1, LEAF_C1 = polygon_pixel_window(poly_geo, src=_ds_b5)
_ds_b5 = None
WIN_R = LEAF_R1 - LEAF_R0 + 1
WIN_C = LEAF_C1 - LEAF_C0 + 1
print(f"Flona window: {WIN_R} rows x {WIN_C} cols "
      f"(r={LEAF_R0}..{LEAF_R1}, c={LEAF_C0}..{LEAF_C1})")

# I/O helpers 
def read_band(path, r0, c0, r1, c1):
    ds = gdal.Open(path)
    arr = ds.GetRasterBand(1).ReadAsArray(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
    ds = None
    return arr

def dn_to_refl(dn):
    r = dn.astype(numpy.float32) * SR + SO
    r[dn == 0] = numpy.nan
    return r

#load reflectance
print("\nReading bands ...")
refl = {}
for date in (DATE_PRE, DATE_POST):
    refl[date] = {b: dn_to_refl(read_band(BAND_FILES[date][b],
                                         LEAF_R0, LEAF_C0, LEAF_R1, LEAF_C1))
                  for b in REQUIRED_BANDS}
    print(f"  {date}: { {b: refl[date][b].shape for b in refl[date]} }")

# index formulas
def idx_nbr(rr, r1):     return (rr - r1) / (rr + r1 + 1e-9)
def idx_bais2(rr, sw2):  return (rr - sw2) / (rr + sw2 + 1e-9)
def idx_ndvi(nr, rd):    return (nr - rd) / (nr + rd + 1e-9)
def idx_evi(nr, rd, bd, g=2.5, c1=6.0, c2=7.5, L=1.0):
    return g * (nr - rd) / (nr + c1 * rd - c2 * bd + L)
def idx_savi(nr, rd, L=0.5): return (1 + L) * (nr - rd) / (nr + rd + L)
def idx_frp(sw2, nr):        return sw2 / (nr + 1.0)
def idx_bai(sw1, nr):        return (sw1 - nr) / (sw1 + nr + 1e-9)
def idx_nbr2(nr, sw2):       return (nr**2 - sw2**2) / (nr**2 + sw2**2 + 1e-9)

# aliases (readability); "21"/"29" suffixes kept for classifier compatibility
pre, post = refl[DATE_PRE], refl[DATE_POST]
print("Computing indices ...")
IDX = {
    "NBR_early":  idx_nbr (pre ["B5"], pre ["B6"]),
    "NBR_late":  idx_nbr (post["B5"], post["B6"]),
    "dNBR":    idx_nbr (pre ["B5"], pre ["B6"]) - idx_nbr(post["B5"], post["B6"]),
    "BAIS2":   idx_bais2(pre ["B5"], pre ["B7"]),
    "BAIS2_late":idx_bais2(post["B5"], post["B7"]),
    "NDVI_early": idx_ndvi (pre ["B5"], pre ["B4"]),
    "NDVI_late": idx_ndvi (post["B5"], post["B4"]),
    "EVI_early":  idx_evi  (pre ["B5"], pre ["B4"], pre ["B2"]),
    "EVI_late":  idx_evi  (post["B5"], post["B4"], post["B2"]),
    "SAVI_early": idx_savi (pre ["B5"], pre ["B4"]),
    "SAVI_late": idx_savi (post["B5"], post["B4"]),
    "FRP_early":  idx_frp  (pre ["B7"], pre ["B5"]),
    "FRP_late":  idx_frp  (post["B7"], post["B5"]),
    "BAI_early":  idx_bai  (pre ["B6"], pre ["B5"]),
    "BAI_late":  idx_bai  (post["B6"], post["B5"]),
    "NBR2_early": idx_nbr2 (pre ["B5"], pre ["B7"]),
    "NBR2_late": idx_nbr2 (post["B5"], post["B7"]),
}

# build DataFrame 
print("Building dataframe ...")
rows = []
for i in range(WIN_R):
    for j in range(WIN_C):
        d = {
            "row": LEAF_R0 + i, "col": LEAF_C0 + j,
            "red_early": pre ["B4"][i, j], "nir21": pre ["B5"][i, j],
            "swir1_early": pre ["B6"][i, j], "swir2_21": pre ["B7"][i, j],
            "b2_early": pre ["B2"][i, j],
            "red_late": post["B4"][i, j], "nir29": post["B5"][i, j],
            "swir1_late": post["B6"][i, j], "swir2_29": post["B7"][i, j],
            "b2_late": post["B2"][i, j],
        }
        d.update({k: v[i, j] for k, v in IDX.items()})
        rows.append(d)

df = pandas.DataFrame(rows)
CSV = os.path.join(DAD, "pixels.csv")
df.to_csv(CSV, index=False, float_format="%.7f")
print(f"\nSaved {len(df)} pixels -> {CSV}")

# stats & histogramas 
print("\nStatistics (Flona window):")
print(df.describe().round(4).to_string())

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
cols_hist = ["NBR_early", "NBR_late", "dNBR", "BAIS2", "EVI_early", "SAVI_early",
             "FRP_early", "BAI_early", "NBR2_early"]
for ax, col in zip(axes.flat, cols_hist):
    ax.hist(df[col].dropna(), bins=40, color="steelblue", alpha=0.75)
    ax.set_title(col); ax.set_xlabel(col); ax.set_ylabel("Freq")
fig.suptitle(f"Flona window — index distributions ({DATE_PRE} vs {DATE_POST})")
fig.tight_layout()
fig.savefig(os.path.join(FIGR, "indices_histograma.png"), dpi=150)
plt.close(fig)
print("Saved -> figuras/indices_histograma.png")