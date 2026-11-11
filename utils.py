import os
import numpy as np
import shapefile
from shapely.geometry import Polygon as ShapelyPolygon, Point
from shapely.ops import unary_union
from pyproj import Transformer
from osgeo import gdal

# Caminho do raster derivado de __file__ (mesma convenção do classifier.py)
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT   = os.path.dirname(os.path.dirname(_UTILS_DIR))   # raiz Trabalho_flona
_B5_PATH   = os.path.join(
    _PROJECT, "recuperacao_queimadas", "dados", "21.12.22",
    "LC09_L2SP_221071_20221221_20230317_02_T2_SR_B5.TIF",
)

_TO_utm = Transformer.from_crs("EPSG:4326", "EPSG:32623", always_xy=True)


def get_gt(src):
    """Devolve (ox, oy, px, py) de um dataset GDAL."""
    gt = src.GetGeoTransform()
    return gt[0], gt[3], gt[1], gt[5]


def _geotransform(src=None):
    """(ox, oy, px, py) — de `src` se dado, senão do B5.TIF. Falha claro se
    o arquivo não existir."""
    if src is None:
        src = gdal.Open(_B5_PATH, gdal.GA_ReadOnly)
        if src is None:
            raise FileNotFoundError(f"Raster B5 não encontrado: {_B5_PATH}")
    return get_gt(src)


def load_flona_polygon(shp_path):
    r = shapefile.Reader(shp_path)
    parts = [ShapelyPolygon(p.points) for p in r.shapes()]
    return unary_union(parts) if len(parts) > 1 else parts[0]


def polygon_utm_bounds(polygon_geo):
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for x, y in polygon_geo.exterior.coords:
        ex, en = _TO_utm.transform(x, y)
        minx = min(minx, ex); miny = min(miny, en)
        maxx = max(maxx, ex); maxy = max(maxy, en)
    return minx, miny, maxx, maxy


def polygon_pixel_window(polygon_geo, src=None):
    """(r0, c0, r1, c1) inclusivo da bbox do polígono na grade da imagem."""
    ox, oy, px, py = _geotransform(src)
    minx, miny, maxx, maxy = polygon_utm_bounds(polygon_geo)
    # py é negativo (north-up): row = (northing - oy) / py
    r0 = int(round((maxy - oy) / py))
    r1 = int(round((miny - oy) / py))
    c0 = int(round((minx - ox) / px))
    c1 = int(round((maxx - ox) / px))
    return r0, c0, r1, c1


def build_mask(polygon_geo, src=None):
    ox, oy, px, py = _geotransform(src)
    r0, c0, r1, c1 = polygon_pixel_window(polygon_geo, src)
    rows = np.arange(r0, r1 + 1)
    cols = np.arange(c0, c1 + 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    easting  = ox + (cc + 0.5) * px
    northing = oy + (rr + 0.5) * py
    poly_utm = ShapelyPolygon(
        [_TO_utm.transform(x, y) for x, y in polygon_geo.exterior.coords]
    )
    inside = np.array([
        poly_utm.contains(Point(e, n))
        for e, n in zip(easting.ravel(), northing.ravel())
    ]).reshape(rr.shape)
    mask = np.zeros((IMG_ROWS, IMG_COLS), dtype=bool)
    mask[r0:r1 + 1, c0:c1 + 1] = inside
    return mask