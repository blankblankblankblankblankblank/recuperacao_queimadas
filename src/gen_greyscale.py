import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pega a pasta do script
PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
# pega a pasta acima (recuperacao_queimadas)
PROJECT = os.path.dirname(PASTA_SCRIPT)
FIG1 = os.path.join(PROJECT, "figuras")
FIGR = os.path.join(FIG1, "indices")
os.makedirs(FIGR, exist_ok=True)

CSV = os.path.join(PROJECT, "dados", "pixels.csv")
df = pd.read_csv(CSV)
print("Loaded", len(df), "pixels")

INDICATORS = [c for c in df.columns if c not in ("row", "col")]
print("Indicators:", INDICATORS)

print("row range:", df["row"].min(), "-", df["row"].max())
print("col range:", df["col"].min(), "-", df["col"].max())
nrows = int(df["row"].max() - df["row"].min() + 1)
ncols = int(df["col"].max() - df["col"].min() + 1)
print("window:", nrows, "x", ncols)

def pivot_indicator(series):
    s = series.astype(float)
    mat = np.full((nrows, ncols), np.nan, dtype=np.float32)
    r = (df["row"].values - df["row"].min()).astype(int)
    c = (df["col"].values - df["col"].min()).astype(int)
    mat[r, c] = s.values
    return mat

for col in INDICATORS:
    print("Processing", col)
    mat = pivot_indicator(df[col])
    out = os.path.join(FIGR, f"greyscale_{col}.png")
    fig, ax = plt.subplots(figsize=(8, 7.5))
    im = ax.imshow(mat, cmap="gray", interpolation="nearest")
    ax.set_title(f"{col} (grayscale)", fontsize=11)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("value")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   ->", out)

print("DONE")
