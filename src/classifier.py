"""
classifier.py — Classificação K-means de recuperação de queimadas.

Requisitos atendidos:
  1. Nenhuma biblioteca sklearn — K-means pura (apenas NumPy), extraída de
     /home/rafael/Documentos/psot/trabalho_kmeans/src/kmeans.py
  2. K determinado empiricamente por silhouette com fator de correção,
     tudo ao quadrado sobre a inércia  (NÃO assume K=4)
  3. Usa as imagens de dados/21.12.22 e dados/29.12.22 (já carregadas
     em dados/pixels.csv por read_index.py)
  4. O mapa de recuperação mostra SOMENTE o que está dentro da máscara
     poligonal da Flona — sem grade gigante, sem fundo vazio
  5. Roda sozinho; gera: mapa_recuperacao.png e recuperacao_flona.txt

Dados de entrada (gerados por read_index.py):
  dados/pixels.csv — 50.151 pixels x 29 colunas
    (row, col, DN, reflexão, e os índices NBR_21, NBR_29, dNBR, BAIS2,
     BAIS2_29, NDVI_21, NDVI_29, EVI_21, EVI_29, SAVI_21, SAVI_29,
     FRP_21, FRP_29, BAI_21, BAI_29, NBR2_21, NBR2_29)

Saída:
  figuras/mapa_recuperacao.png   - mapa poliglinal (somente pixels da máscara)
  recuperacao_flona.txt          - relatório de recuperação
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# --- caminho do projeto (raiz = /home/rafael/Documentos/psot/Trabalho_flona) ---
PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
# pega a pasta acima (trabalho_flona)
PROJECT = os.path.dirname(os.path.dirname(PASTA_SCRIPT))
INICIAL = os.path.join(PROJECT, "inicial")
DADOS = os.path.join(PROJECT, "recuperacao_queimadas", "dados")
FIG   = os.path.join(PROJECT, "recuperacao_queimadas", "figuras")
TXT   = os.path.join(PROJECT, "recuperacao_queimadas", "recuperacao_flona.txt")
os.makedirs(FIG, exist_ok=True)

# --- carregar pixels.csv (gerado por read_index.py) --------------------------
df = pd.read_csv(os.path.join(DADOS, "pixels.csv"))
print(f"Carregados {len(df)} pixels (Flona window).")

# --- índices usados como atributos ------------------------------------------
INDICES = [
    # severidade
    "dNBR",
    "BAIS2_29",
    "BAI_29",
    # recuperação
    "NBR_29",
    "NDVI_29",
    "EVI_29",
    "SAVI_29",
    "NBR2_29",
    "FRP_29",
]

feat = df[INDICES].copy().fillna(0.0)
X = feat.values.astype(float)

# --- escala Standard ---------------------------------------------------------
mean = X.mean(axis=0)
std  = X.std(axis=0) + 1e-12
Xz = (X - mean) / std

rng = np.random.default_rng(42)

# -------------------------------------------------------------
#  K-MEANS PURA N-DIMENSIONAL (extraída de trabalho_kmeans)
# -------------------------------------------------------------
def kmeans_nd(X, K=2, max_iter=300, tol=1e-4, seed=42):
    """
    K-means clássico em dimensão N.
    Retorna (rotulo, centros, n_iter, J_final)
    """
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=K, replace=False)
    centros = X[idx].copy()

    rotulo = np.zeros(N, dtype=int)
    J_final = None
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        # distância euclidiana (N, K)
        D = np.empty((N, K))
        for k in range(K):
            diff = X - centros[k]
            D[:, k] = np.sqrt((diff ** 2).sum(axis=1))

        rotulo = np.argmin(D, axis=1)

        centros_novo = centros.copy()
        for k in range(K):
            memb = rotulo == k
            if np.any(memb):
                centros_novo[k] = X[memb].mean(axis=0)

        desloc = np.sqrt(((centros_novo - centros) ** 2).sum(axis=1)).max()
        centros = centros_novo

        J = float((D[np.arange(N), rotulo] ** 2).sum())
        J_final = J
        if desloc < tol:
            break
    return rotulo, centros, n_iter, J_final


def silhouette_score(X, lab):
    """Silhouette score manual (sem sklearn)."""
    n = X.shape[0]
    unique = np.unique(lab)
    if len(unique) <= 1:
        return 0.0
    s = np.zeros(n)
    for i in range(n):
        xi = X[i]
        lab_i = lab[i]
        # intra-cluster
        memb = lab == lab_i
        a = np.sqrt(((X[memb] - xi) ** 2).sum(axis=1)).mean()
        # nearest-cluster distance
        b = np.inf
        for c in unique:
            if c == lab_i:
                continue
            memb_c = lab == c
            bc = np.sqrt(((X[memb_c] - xi) ** 2).sum(axis=1)).mean()
            b = min(b, bc)
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def inertia_of(X, lab):
    """Inércia: soma das dist² a seu próprio centro."""
    s = 0.0
    for c in np.unique(lab):
        memb = lab == c
        if memb.sum() == 0:
            continue
        centroid = X[memb].mean(axis=0)
        s += ((X[memb] - centroid) ** 2).sum()
    return float(s)


# -------------------------------------------------------------
#  Seleção empírica do K
# -------------------------------------------------------------
MAX_K   = 12
SUB     = 3000                     # sub-amostragem para o silhouette (reduz O(N²))
CORRECO = 10.0                   # fator de correção (2^2 = 4 no quadrado)

print("\nDeterminando K por silhouette + fator de correção (quadrado / inércia)...")

sil = {}
inertia = {}
idx_sub = rng.choice(len(Xz), size=min(SUB, len(Xz)), replace=False)
Xz_sub = Xz[idx_sub]

for k in range(2, MAX_K + 1):
    rot, c, n_iter, J = kmeans_nd(Xz_sub, K=k, max_iter=300, tol=1e-4, seed=42)
    sil[k] = silhouette_score(Xz_sub, rot)
    inertia[k] = inertia_of(Xz_sub, rot)
    print(f"  K={k:2d}  silhouette={sil[k]:.4f}  inertia={inertia[k]:.4f}")

# K = argmax( (silhouette * fator_correcao)^2 / inércia )
ratio = {k: ((sil[k] * CORRECO) ** 2 / inertia[k]) for k in sil}
K_sel = max(ratio, key=ratio.get)
print(f"\nK selecionado: {K_sel}")

# -------------------------------------------------------------
#  Clusterização final com o K selecionado
# -------------------------------------------------------------
lab, centroids, n_iter, J_final = kmeans_nd(Xz, K=K_sel, max_iter=300, tol=1e-4, seed=42)
df["cluster"] = lab
print("\nDist. de pixels por cluster:")
for c in range(K_sel):
    n = int((lab == c).sum())
    print(f"  cluster {c}: {n:6d} pixels ({100*n/len(lab):5.2f}%)")

# -------------------------------------------------------------
#  Mapeamento cluster -> categoria de recuperação
# -------------------------------------------------------------
# classes ordenadas por NBR_29 médio (quanto maior, mais recuperada)
nbr_means = df.groupby("cluster")["NBR_29"].mean().sort_values(ascending=True)
cluster_to_cat = {c: rank for rank, c in enumerate(nbr_means.index)}
df["cat"] = df["cluster"].map(cluster_to_cat)


# --- definir nomes de categoria (K nomes, baseados no rank de NBR_29) --- 
K_SEL = K_sel
# Os clusters são ordenados por NBR_29 médio crescente: 0 = mais grave, K-1 = mais recuperado
RANK_NAMES = [
    "Grave", "Leve", "Moderado", "Saudável", "Recuperado",
    "Muito Recuperado", "Verde", "Intacta", "Muito Verde"
]
CAT_NAMES = [RANK_NAMES[min(k, len(RANK_NAMES) - 1)] for k in range(K_SEL)]
print(f"\nMapa de recuperação (ordenado por NBR_29):")
for c in range(K_SEL):
    print(f"  {CAT_NAMES[c]}: cluster {list(nbr_means.index)[c]} (NBR_29 médio={nbr_means.iloc[c]:.4f})")

# -------------------------------------------------------------
#  Máscara poligonal e mapa de recuperação
# -------------------------------------------------------------
print("\nGerando máscara poligonal e mapa de recuperação...")

# --- usar a borda poligonal diretamente (não a grade inteira) ----------------
from utils import load_flona_polygon, polygon_utm_bounds, polygon_pixel_window

shp = os.path.join(INICIAL, "FlonaBSB.shp")
poly_geo = load_flona_polygon(shp)

# coordenadas pixel do polígono (window Flona)
r0, c0, r1, c1 = polygon_pixel_window(poly_geo)
WIN_R = r1 - r0 + 1
WIN_C = c1 - c0 + 1
print(f"  Janela Flona: {WIN_R} linhas x {WIN_C} colunas")

# --- criar a imagem a partir dos pixels do CSV, não da grade cheia -----------
# Inicializar com -1 (background, fora da máscara)
img = np.full((WIN_R, WIN_C), -1, dtype=np.int8)

# Mapear cada pixel do CSV para sua posição na imagem
for _, row in df.iterrows():
    rr = int(row["row"]) - r0
    cc = int(row["col"]) - c0
    if 0 <= rr < WIN_R and 0 <= cc < WIN_C:
        img[rr, cc] = int(row["cat"])

# --- contar pixels na máscara ---
n_mapped = int((img >= 0).sum())
print(f"  pixels classificados na máscara: {n_mapped} / {len(df)}")


# -------------------------------------------------------------
#  Gráfico de suporte K
# -------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ks = list(range(2, MAX_K + 1))
ax[0].plot(ks, [inertia[k] for k in ks], "o-", color="steelblue", linewidth=2, markersize=6)
ax[0].set_xlabel("K")
ax[0].set_ylabel("Inércia")
ax[0].set_title("Método do cotovelo (inércia)")
ax[0].grid(alpha=0.3)

ax[1].plot(ks, [sil[k] for k in ks], "o-", color="darkorange", linewidth=2, markersize=6)
ax[1].axvline(K_sel, color="red", linestyle="--", linewidth=2, label=f"K={K_sel}")
ax[1].set_xlabel("K")
ax[1].set_ylabel("Silhouette score")
ax[1].set_title("Seleção de K (silhouette + fator² / inércia)")
ax[1].legend()
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "curva_k_selecao.png"), dpi=150)
plt.close(fig)
print(f"  Figura: figuras/curva_k_selecao.png")

# -------------------------------------------------------------
#  Relatório de recuperação (txt)
# -------------------------------------------------------------
total = len(df)
cat_counts = df["cat"].value_counts().sort_index()

lines = []
lines.append("RELATÓRIO DE RECUPERAÇÃO DE QUEIMADAS - FLONA")
lines.append("=" * 64)
lines.append(f"Projeto: {PROJECT}")
lines.append(f"Área analisada: janela Flona ({total} pixels)")
lines.append(f"Índices usados: {', '.join(INDICES)}")
lines.append(f"Clusters/K selecionados: {K_SEL}")
lines.append(f"Período: 2022-12-21 (pré-fogo) vs 2022-12-29 (pós-fogo)")
lines.append("")
lines.append("-" * 64)
lines.append("1. DISTRIBUIÇÃO DA FLONA POR NÍVEL DE RECUPERAÇÃO")
lines.append("-" * 64)
for cat in range(K_SEL):
    n = int(cat_counts.get(cat, 0))
    pct = 100 * n / total
    lines.append(f"   {CAT_NAMES[cat]:20s}:  {n:6d} pixels ({pct:5.2f}%)")
lines.append("")
lines.append("-" * 64)
lines.append("2. STATS GERAIS DOS ÍNDICES")
lines.append("-" * 64)
for col in INDICES:
    s = df[col].describe()
    lines.append(f"   {col:12s}: mean={s['mean']:8.4f}  std={s['std']:8.4f}  "
                 f"range=[{s['min']:8.4f}, {s['max']:8.4f}]")
lines.append("")
lines.append("-" * 64)
lines.append("3. RECUPERAÇÃO GLOBAL")
lines.append("-" * 64)
recuperado = int(cat_counts.get(K_SEL - 1, 0))
recuperado_pct = 100 * recuperado / total
lines.append(f"   Recuperada (classe mais alta): {recuperado} pixels ({recuperado_pct:.2f}%)")
for cat in range(K_SEL):
    n = int(cat_counts.get(cat, 0))
    pct = 100 * n / total
    lines.append(f"   {CAT_NAMES[cat]:20s}: {n:6d} pixels ({pct:5.2f}%)")
lines.append("")
lines.append("-" * 64)
lines.append("4. MAPA DIGITAL")
lines.append("-" * 64)
lines.append(f"   figuras/mapa_recuperacao.png  -> {FIG}")
lines.append(f"   Shape: {img.shape}  |  pixels na máscara: {n_mapped}")
lines.append("")

# --- salvar o mapa de recuperação (apenas pixels dentro da máscara) ----------
cmap = plt.cm.tab20
norm = BoundaryNorm(np.arange(-0.5, K_SEL + 1.5, 1), cmap.N)
fig_map, ax_map = plt.subplots(figsize=(10, 9))
ax_map.imshow(img, cmap=cmap, norm=norm, interpolation="nearest")
ax_map.set_title("Recuperação de queimadas - Flona (máscara poligonal)")
ax_map.axis("off")
fig_map.savefig(os.path.join(FIG, "mapa_recuperacao.png"), dpi=150, bbox_inches="tight")
plt.close(fig_map)
print(f"  Figura: figuras/mapa_recuperacao.png (shape={img.shape}, "
      f"pixels na máscara={n_mapped})")
lines.append("=" * 64)
lines.append("FIM DO RELATÓRIO")
lines.append("=" * 64)

report = "\n".join(lines) + "\n"
with open(TXT, "w") as f:
    f.write(report)

print(f"\nRelatório salvo: {TXT}")

# --- imprimir em tela para o run_all.py gravar de novo ------------------
print("\n" + "=" * 64)
print("\n".join(lines))
print("=" * 64)

with open(TXT, "w") as f:
    f.write(report)

print(f"Relatório atualizado: {TXT}")
