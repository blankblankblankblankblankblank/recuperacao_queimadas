"""
Make histograms of the index distributions from the already-computed CSV.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGR = os.path.join(PROJECT, "recuperacao_queimadas", "figuras")
os.makedirs(FIGR, exist_ok=True)

CSV = os.path.join(PROJECT, "recuperacao_queimadas", "dados", "pixels.csv")
df = pd.read_csv(CSV)
print(f"Loaded {len(df)} rows from {CSV}")

cols_hist = ["NBR_21", "NBR_29", "dNBR", "BAIS2", "EVI_21", "SAVI_21",
             "FRP_21", "BAI_21", "NBR2_21"]

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
for ax, col in zip(axes.flat, cols_hist):
    ax.hist(df[col].dropna(), bins=40, color="steelblue", alpha=0.75)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Freq")
fig.suptitle("Flona window – index distributions (21.12.22 vs 29.12.22)", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIGR, "indices_histograma.png"), dpi=150)
plt.close(fig)
print("Saved -> figuras/indices_histograma.png")

# also a 2-panel dNBR vs NBR_21 scatter (like trabalho_kmeans's dispersao.png)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(df["NBR_21"], df["dNBR"], c=df["EVI_21"], cmap="YlGn", s=3, alpha=0.4)
axes[0].set_xlabel("NBR_21")
axes[0].set_ylabel("dNBR")
axes[0].set_title("NBR_21 vs dNBR (colour = EVI_21)")

axes[1].scatter(df["NBR_21"], df["BAIS2"], c=df["EVI_21"], cmap="YlGn", s=3, alpha=0.4)
axes[1].set_xlabel("NBR_21")
axes[1].set_ylabel("BAIS2")
axes[1].set_title("NBR_21 vs BAIS2 (colour = EVI_21)")
fig.tight_layout()
fig.savefig(os.path.join(FIGR, "dispersao_indices.png"), dpi=150)
plt.close(fig)
print("Saved -> figuras/dispersao_indices.png")
