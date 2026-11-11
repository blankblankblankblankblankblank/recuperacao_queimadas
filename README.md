# Recuperação de Queimadas — Flona

Pipeline de classificação não supervisionada (K-means, **sem scikit-learn**)
para avaliar a recuperação de áreas queimadas na Floresta Nacional de Brasília
(Flona), a partir de imagens Landsat C2 L2 em duas datas (pré e pós-fogo).

O pipeline calcula índices espectrais de severidade de queimada e de
recuperação de vegetação, determina **empiricamente** o número de clusters K,
classifica cada pixel da Flona e gera relatório, mapa e imagens de suporte.

---

## 1. Estrutura do projeto

```
Trabalho_flona/
├── inicial/
│   └── FlonaBSB.shp                 # polígono da Flona (WGS84, EPSG:4326)
└── recuperacao_queimadas/
    ├── dados/
    │   ├── 21.12.22/                # cena pré-fogo (pasta = data, dd.mm.yy)
    │   │   └── *SR_B2.TIF … *SR_B7.TIF
    │   ├── 29.12.22/                # cena pós-fogo
    │   │   └── *SR_B2.TIF … *SR_B7.TIF
    │   └── pixels.csv               # gerado por read_index.py
    ├── figuras/
    │   ├── indices_histograma.png
    │   ├── curva_k_selecao.png
    │   ├── mapa_recuperacao.png
    │   ├── dispersao_indices.png
    │   └── indices/
    │       └── greyscale_*.png      # um por indicador
    ├── src/
    │   ├── run_all.py               # orquestrador
    │   ├── read_index.py            # step 1 — leitura + índices
    │   ├── classifier.py            # step 2 — K-means + mapa + relatório
    │   ├── make_histograms.py       # step 3 — histogramas adicionais
    │   ├── gen_greyscale.py         # step 4 — imagens grayscale
    │   ├── kmeans.py                # implementação pura (NumPy)
    │   ├── index_utils.py           # fórmulas dos índices
    │   └── utils.py                 # geometria, máscara, geotransform
    └── recuperacao_flona.txt        # relatório final
```

---

## 2. Requisitos

- **Python 3.8+**
- **GDAL** (`gdal-bin`, `python3-gdal`) — obrigatório para ler os `.TIF`
- Pacotes Python:

  ```bash
  pip install numpy pandas matplotlib pyshp shapely pyproj
  ```

  No Ubuntu/Debian, prefira os pacotes do sistema:
  ```bash
  sudo apt install python3-numpy python3-pandas python3-matplotlib \
                   python3-shapefile python3-shapely python3-pyproj \
                   python3-gdal
  ```

**Não** é necessário scikit-learn — o K-means está implementado em `src/kmeans.py`.

---

## 3. Como rodar

### Pipeline completo (recomendado)

```bash
cd recuperacao_queimadas/src
python3 run_all.py
```

Executa os 4 steps em sequência e imprime um resumo:

```
RESUMO DO PIPELINE
  [OK  ] read_index.py
  [OK  ] classifier.py
  [OK  ] make_histograms.py
  [OK  ] gen_greyscale.py
  4/4 scripts executados com sucesso.
```

Exit code `0` se tudo passou, `1` se algum step falhou.

### Step a step (debug)

```bash
cd recuperacao_queimadas/src
python3 read_index.py        # gera dados/pixels.csv
python3 classifier.py        # gera mapa + relatório
python3 make_histograms.py
python3 gen_greyscale.py
```

---

## 4. Convenção de pastas de cena — **crítico**

O `read_index.py` descobre as cenas sozinho, sem hardcode de data. A regra:

> **Cada subpasta de `dados/` cujo nome siga `dd.mm.yy` (ou `dd.mm.yyyy`)
> é uma cena. A mais antiga vira `early` (pré-fogo); a mais nova, `late`
> (pós-fogo). O nome dos TIFs não importa, só os sufixos `_SR_B{2,4,5,6,7}.TIF`.**

Exemplo: para adicionar uma terceira data em janeiro de 2023:

```
dados/
├── 21.12.22/     ← early
├── 29.12.22/     ← late (atual)
└── 05.01.23/     ← se criada, passa a ser late (a mais nova)
```

Basta criar a pasta e soltar os TIFs. **Não é preciso editar código.**

Bandas obrigatórias por cena: **B2, B4, B5, B6, B7** (blue, red, NIR, SWIR1,
SWIR2). Se faltar alguma, o pipeline falha com mensagem clara.

---

## 5. Pipeline — o que cada step faz

### Step 1 — `read_index.py`

1. Descobre as cenas em `dados/` (convenção `dd.mm.yy`, ver §4).
2. Lê as bandas da janela da Flona — a bbox do polígono em `inicial/FlonaBSB.shp`
   projetada em UTM 23S / EPSG:32623, via `utils.polygon_pixel_window`.
3. Converte DN → reflectância de superfície (escala C2 L2: `0.0000275 · DN − 0.2`).
4. Calcula os índices espectrais:

   | Índice | Fórmula | Uso |
   |---|---|---|
   | **NBR** | `(NIR − SWIR2) / (NIR + SWIR2)` | severidade de queimada |
   | **dNBR** | `NBR_early − NBR_late` | mudança entre datas |
   | **BAIS2** | `(NIR − SWIR2) / (NIR + SWIR2)` (2 bandas) | área queimada |
   | **NDVI** | `(NIR − Red) / (NIR + Red)` | vigor vegetal |
   | **EVI** | `2.5·(NIR − Red) / (NIR + 6·Red − 7.5·Blue + 1)` | vegetação densa |
   | **SAVI** | `1.5·(NIR − Red) / (NIR + Red + 0.5)` | solo exposto |
   | **FRP** | `SWIR2 / (NIR + 1)` | proxy de fogo ativo |
   | **BAI** | `(SWIR1 − NIR) / (SWIR1 + NIR)` | cinza queimada |
   | **NBR2** | `(NIR² − SWIR2²) / (NIR² + SWIR2²)` | severidade complementar |

5. Exporta `dados/pixels.csv` — uma linha por pixel, com `row`, `col`,
   reflectâncias e índices (colunas com sufixo `_early`/`_late`).
6. Gera `figuras/indices_histograma.png`.

### Step 2 — `classifier.py`

1. Carrega `pixels.csv` e escala as features (StandardScaler).
2. **Seleciona K empiricamente** — para cada `K ∈ [2, 12]`, roda K-means numa
   subamostra de 3.000 pixels e calcula:
   - Silhouette score (implementação manual, O(N²))
   - Inércia (soma das dist² ao centróide)

   Escolhe `K = argmax( (silhouette · 10)² / inércia )`.
3. Clusteriza **todos** os pixels com o K escolhido.
4. Ordena os clusters por `NBR_late` médio → rank 0 = mais grave, K−1 = mais
   recuperado.
5. Atribui nomes (`Grave`, `Leve`, `Moderado`, … — lista fixa por K, §7).
6. Reconstrói o mapa a partir de `(row, col)` de cada pixel → matriz
   `img[WIN_R, WIN_C]` com `−1` fora da máscara poligonal.
7. Gera:
   - `figuras/curva_k_selecao.png` (cotovelo + silhouette + K escolhido)
   - `figuras/mapa_recuperacao.png` (mapa + legenda discreta à direita)
   - `recuperacao_flona.txt` (relatório)

### Step 3 — `make_histograms.py`

Histogramas complementares (`indices_histograma.png` expandido) e scatter
`NBR_early × dNBR` + `NBR_early × BAIS2` (`dispersao_indices.png`).

### Step 4 — `gen_greyscale.py`

Gera um PNG grayscale por indicador em `figuras/indices/greyscale_<indicador>.png`
(útil para inspeção visual pixel-a-pixel).

---

## 6. Saídas

| Arquivo | Descrição |
|---|---|
| `dados/pixels.csv` | tabela completa — uma linha por pixel da janela Flona |
| `figuras/indices_histograma.png` | distribuições dos índices |
| `figuras/curva_k_selecao.png` | curva de cotovelo + silhouette + K escolhido |
| `figuras/mapa_recuperacao.png` | mapa classificado (somente pixels da máscara) |
| `figuras/dispersao_indices.png` | scatter `NBR_early × dNBR` e `× BAIS2` |
| `figuras/indices/greyscale_*.png` | uma imagem grayscale por indicador |
| `recuperacao_flona.txt` | relatório final (distribuição, stats, recuperação global) |

---

## 7. Decisões de projeto

- **Sem scikit-learn.** K-means em `src/kmeans.py` (`kmeans_nd`), silhouette e
  inércia implementados à mão. Só NumPy.
- **K não é assumido.** É escolhido empiricamente pelo critério
  `(silhouette · fator)² / inércia`. O fator `10` (constante `CORRECO`) foi
  calibrado para este dataset — ajuste em `classifier.py` se for reutilizar em
  outro contexto.
- **Nomes de categoria por K.** Lista `NAMES_BY_K` em `classifier.py`; para K
  fora dela, cai em `Nível i/N`.
- **`early`/`late` pela data real da pasta**, não pela ordem alfabética. Um
  `02.01.23` ordena corretamente depois de `29.12.22` (via `datetime.date`).
- **Colunas do CSV:** `_early` (pré-fogo) e `_late` (pós-fogo). As colunas
  antigas `_21`/`_29` **não existem mais** — se você tem um `pixels.csv` antigo,
  apague-o antes de rodar.
- **Sem caminhos hardcoded.** Todos os caminhos derivam de `__file__`; a única
  exceção é o fallback de `utils._geotransform` quando chamado sem `src=` — mas
  `read_index.py` e `classifier.py` sempre passam `src` explicitamente.

---

## 8. Troubleshooting

### `SystemExit: 1` no Spyder

É o `run_all.py` reportando “algum step falhou” (código de saída `1`). Não é o
erro em si — role o output para cima e procure as linhas:

```
  FALHA no script <nome> (exit N)
  STDERR:
   ...
```

O `STDERR` indica o problema real (ex.: `KeyError: 'NBR_29'` se o CSV for do
run antigo).

### `KeyError: 'NBR_29'` / `KeyError: 'NBR_21'`

`pixels.csv` ainda está no formato antigo (`_21`/`_29`). Apague e regenere:

```bash
rm dados/pixels.csv
python3 src/read_index.py
```

### `FileNotFoundError: Raster B5 não encontrado: /home/…`

Se você rodar `classifier.py` **sozinho** e ele cair no `utils._geotransform`
sem `src=`, o `utils.py` tenta um caminho fixo. O pipeline (`run_all.py`) não
tem esse problema porque `read_index.py` passa `src` explicitamente. Se quiser
rodar `classifier.py` isolado, edite o `_B5_PATH` em `utils.py` ou passe
`src=` (veja o exemplo em `read_index.py`).

### `ERROR 4: … .TIF: No such file or directory` + `AttributeError: 'NoneType'`

`gdal.Open` retornou `None` (arquivo ausente) e o código antigo continuou
chamando `.GetGeoTransform()`. Verifique se os TIFs da cena existem e se as
pastas seguem `dd.mm.yy`. O `utils.py` atual já levanta `FileNotFoundError`
com o caminho, em vez de estourar `AttributeError`.

### `Preciso de >=2 pastas de data em …`

`read_index.py` só achou 0 ou 1 cena. Confira:

- Pastas em `dados/` nomeadas `dd.mm.yy` (com **pontos**, não hífens nem underscores)
- TIFs dentro de cada pasta com sufixo `_SR_B*.TIF`
- Permissão de leitura

### A linha vermelha do K escolhido sumiu no gráfico

Bug conhecido: `axvline(..., linestyle="")` desenha linha vazia. Use `"--"`.

---

## 9. Notas técnicas

### Geotransform

Imagens Landsat C2 L2, WGS84 / UTM zone 23N (EPSG:32623). Geotransform
north-up, pixel 30 m, origem `(72585, −1645185)`, grade 7681 × 7571.
O polígono da Flona está em EPSG:4326 e é reprojetado em runtime.

### Máscara poligonal

`utils.build_mask` (opcional) faz point-in-polygon pixel a pixel. O
`classifier.py` **não usa** `build_mask` — em vez disso, constrói `img` a
partir dos `(row, col)` do CSV, deixando `−1` (background) para fora da
máscara. É mais rápido e evita carregar a grade inteira.

### Silhouette score

Implementação O(N²), viável apenas na subamostra (`SUB = 3000`). Para o
dataset completo (≈50k pixels), o custo seria ~2,5 G comparações por K.

---

## 10. Licença / autoria

Projeto acadêmico derivado de `trabalho_kmeans`. Dados Landsat de domínio
público (USGS). Polígono da Flona: ICMBio.
