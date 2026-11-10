"""Utils: Flona polygon loading, raster masking and pixel extraction.

Data model (verified with gdalinfo):
  - Images: Landsat C2 L2, WGS84 / UTM zone 23N (EPSG:32623)
  - Geotransform: Origin(73185, -1645185) earlier check, code CONFIRMED
                   origin (72585, -1645185), pixel size 30 m, north-up
  - Grid: 7681 rows x 7571 cols
  - FlonaBSB.shp: geographic WGS84 (EPSG:4326), 27-vertex polygon
"""

import numpy as np
import shapefile
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from pyproj import Transformer
from osgeo import gdal

# ---------------------------------------------------------------------------
# Geo metadata (read dynamically from the actual image at runtime)
# ---------------------------------------------------------------------------
def get_gt(src):
    """Devolve (ox, oy, px, py) de um dataset GDAL."""
    gt = src.GetGeoTransform()
    return gt[0], gt[3], gt[1], gt[5]


def _get_geotransform(src):
    """Return (ox, oy, px, py) from a GDAL dataset, or from the band-5 TIF
    (the primary file for the 21.12.22 scene) when src is None, so the same
    behaviour as the former duplicated branch."""
    if src is not None:
        return get_gt(src)
    p = "/home/rafael/Documentos/psot/Trabalho_flona/recuperacao_queimadas/dados/21.12.22/LC09_L2SP_221071_20221221_20230317_02_T2_SR_B5.TIF"
    src2 = gdal.Open(p)
    gt = src2.GetGeoTransform()
    src2 = None
    return gt[0], gt[3], gt[1], gt[5]


_TO_utm = Transformer.from_crs("EPSG:4326", "EPSG:32623", always_xy=True)


def load_flona_polygon(shp_path):
    """Return a shapely Polygon (WGS84 lon/lat) from FlonaBSB.shp."""
    r = shapefile.Reader(shp_path)
    parts = [ShapelyPolygon(p.points) for p in r.shapes()]
    return unary_union(parts) if len(parts) > 1 else parts[0]


def polygon_utm_bounds(polygon_geo, src=None):
    """Axis-aligned (minX, minY, maxX, maxY) bbox of the polygon in UTM metres."""
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for x, y in polygon_geo.exterior.coords:
        ex, en = _TO_utm.transform(x, y)
        minx = min(minx, ex); miny = min(miny, en)
        maxx = max(maxx, ex); maxy = max(maxy, en)
    return minx, miny, maxx, maxy


def polygon_pixel_window(polygon_geo, src=None):
    """Return (row_min, col_min, row_max, col_max) of the polygon in the
    image grid (inclusive). Reads geotransform from src if provided."""
    gt = src.GetGeoTransform() if src is not None else None
    if gt is None:
        # Recompute for the 21.12.22 file (the first band)
        p = "/home/rafael/Documentos/psot/Trabalho_flona/recuperacao_queimadas/dados/21.12.22/LC09_L2SP_221071_20221221_20230317_02_T2_SR_B5.TIF"
        src2 = gdal.Open(p)
        gt = src2.GetGeoTransform()
        src2 = None
    ox, oy, px, py = gt[0], gt[3], gt[1], gt[5]
    # row = (originY - N) / |px| ; col = (E - originX) / px
    minx, miny, maxx, maxy = polygon_utm_bounds(polygon_geo)
    r0 = int(round((oy - maxy) / px))
    r1 = int(round((oy - miny) / px))
    c0 = int(round((minx - ox) / px))
    c1 = int(round((maxx - ox) / px))
    return r0, c0, r1, c1


def build_mask(polygon_geo, src=None):
    """Boolean (rows, cols) mask of pixels whose centroid lies inside the
    UTM-projected polygon (pixel-centre test)."""
    r0, c0, r1, c1 = polygon_pixel_window(polygon_geo, src)
    gt = src.GetGeoTransform() if src is not None else None
    if gt is None:
        p = "/home/rafael/Documentos/psot/Trabalho_flona/recuperacao_queimadas/dados/21.12.22/LC09_L2SP_221071_20221221_20230317_02_T2_SR_B5.TIF"
        src2 = gdal.Open(p)
        gt = src2.GetGeoTransform()
        src2 = None
    ox, oy, px, py = gt[0], gt[3], gt[1], gt[5]

    # Build pixel grid within the window
    rows = np.arange(r0, r1 + 1)
    cols = np.arange(c0, c1 + 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    # Centroid coordinates in UTM metres
    easting = ox + (cc + 0.5) * px
    northing = oy + (rr + 0.5) * py  # py is negative for north-up

    # UTM polygon
    pts = [_TO_utm.transform(x, y) for x, y in polygon_geo.exterior.coords]
    poly_utm = ShapelyPolygon(pts)

    # Containment test
    from shapely.geometry import Point
    inside = np.array([
        poly_utm.contains(Point(e, n))
        for e, n in zip(easting.ravel(), northing.ravel())
    ]).reshape(rr.shape)

    mask = np.zeros((IMG_ROWS, IMG_COLS), dtype=bool)
    mask[r0:r1 + 1, c0:c1 + 1] = inside
    return mask


# Lazy constants (set once at import time)
class _Ctx:
    IMG_ROWS = 7681
    IMG_COLS = 7571

CTX = _Ctx()
IMG_ROWS = CTX.IMG_ROWS
IMG_COLS = CTX.IMG_COLS
