"""
Bandas:
  B4 = Red     (0.64 - 0.67 um)
  B5 = NIR     (0.85 - 0.88 um)
  B6 = SWIR1   (1.57 - 1.65 um)
  B7 = SWIR2   (2.11 - 2.29 um)
"""

import numpy as np

# Landsat C2 L2 surface reflectance scale / offset
SR_SCALE = 0.0000275
SR_OFFSET = -0.2

# Modal band index names (B4=Red, B5=NIR, B6=SWIR1, B7=SWIR2)
BAND = {
    "blue": 2, "green": 3, "red": 4, "nir": 5,
    "swir1": 6, "swir2": 7,
}


def _to_reflectance(dn):
    """Convert DN to surface reflectance, clip to physically admissible range."""
    rho = dn.astype(np.float64) * SR_SCALE + SR_OFFSET
    return np.clip(rho, 0.0, 1.0)


def _safe_div(num, den):
    """Element-wise division, replace inf/nan with 0."""
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den, out=out, where=(den != 0))
    out[~np.isfinite(out)] = 0.0
    return out


def nbr(red=None, nir=None, swir1=None):
    """Normalized Burn Ratio (Key et al. 2000)."""
    return _safe_div(nir - swir1, nir + swir1)


def nbr2(red=None, nir=None, swir2=None):
    """Normalized Burn Ratio 2 (uses SWIR2)."""
    return _safe_div(nir - swir2, nir + swir2)


def d_nbr(nbr_pre, nbr_post):
    """Differenced NBR: NBR(pre) - NBR(post).  Larger (less negative)
    values indicate less recovery."""
    return nbr_pre - nbr_post


def baisf2(nir=None, swir2=None):
    """BAIS2 (Burned Area Index 2, Filipponi 2018).

    BAIS = (NIR - SWIR2) / (NIR + SWIR2 + NIR + SWIR2)
    Simplified: BAIS = (NIR - SWIR2) / (2*NIR + 2*SWIR2)
    """
    return _safe_div(nir - swir2, 2 * (nir + swir2))


def frp(nir=None, swir2=None, blue=None):
    """Fire Radiative Power proxy (relative).  Uses the band ratio of
    SWIR2 (hot) over a cooler band to flag active fire pixels."""
    # Relative FRP proxy: SWIR2 / (NIR + 1) - emphasises hot pixels
    return _safe_div(swir2, nir + 1.0)


def ndvi(nir=None, red=None):
    """Normalized Difference Vegetation Index."""
    return _safe_div(nir - red, nir + red)


def evi(nir=None, red=None, blue=None, g=2.5, c1=6.0, c2=7.5, L=1.0):
    """Enhanced Vegetation Index (Huete et al. 2002).

    EVI = G * (NIR - RED) / (NIR + C1*RED - C2*BLUE + L)
    """
    num = g * (nir - red)
    den = nir + c1 * red - c2 * blue + L
    return _safe_div(num, den)


def savi(nir=None, red=None, L=0.5):
    """Soil-Adjusted Vegetation Index (Huete 1988).

    SAVI = (NIR - RED) / (NIR + RED + L) * (1 + L)
    """
    num = nir - red
    den = nir + red + L
    return (1 + L) * _safe_div(num, den)


def bai(nir=None, swir1=None):
    """Burned Ash Index (Kearney 2003)."""
    return _safe_div(swir1 - nir, swir1 + nir + 0.000001)


def vcami(nir=None, swir2=None, swir1=None, red=None):
    """Visible Atmospherically Resistant Index (Crist 1985)."""
    return _safe_div(nir - red, nir + red)


def ratio_index(nir=None, swir1=None):
    """Simple reflectance ratio (SWIR1/NIR)."""
    return _safe_div(swir1, nir + 1e-6)
