"""
run_all.py — Pipeline principal de classificação de recuperação de queimadas.

Executa SEQUENCIALMENTE todos os passos do projeto, dentro de:
    /home/rafael/Documentos/psot/Trabalho_flona/recuperacao_queimadas/

Passo 1 : read_index.py   — lê as imagens (21.12.22, 29.12.22), calcula os índices,
                exporta dados/pixels.csv e gera figuras/indices_histograma.png
Step 2  : classifier.py   — determina K empiricamente ( (silhouette x fator)² / inércia),
                classifica os pixels, gera o mapa poliglinal (mascara) e
                recuperacao_flona.txt
Step 3  : make_histograms.py — histogramas adicionais a partir de pixels.csv

Nenhuma biblioteca sklearn é utilizada. A implementação de K-means usa
apenas NumPy, extraída de trabalho_kmeans/src/kmeans.py.

Uso:
    python3 src/run_all.py
"""

import os
import subprocess
import sys


def _has_required_modules(exe):
    #True se `exe` consegue importar numpy, matplotlib e gdal (osgeo).
    try:
        r = subprocess.run(
            [exe, "-c",
             "import numpy, matplotlib; from osgeo import gdal; print('ok')"],
            capture_output=True, text=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def _find_working_python():
    """Retorna um interpretador Python com o stack necessário.

    O interpretador que lançou run_all.py (sys.executable) é testado primeiro;
    se caiu no Hermes venv (Python 3.14 sem numpy) ou estiver deficiente, cai no
    /usr/bin/python3, que possui o stack completo (numpy, matplotlib, gdal).
    """
    candidates = [sys.executable]
    for extra in ("/usr/bin/python3", "/usr/local/bin/python3"):
        if extra not in candidates:
            candidates.append(extra)
    for exe in candidates:
        if exe and os.path.exists(exe) and _has_required_modules(exe):
            return exe
    return sys.executable  # fallback: manter o interpretador atual


# o script é executado de dentro de src/, portanto o root do projeto é
# o parente imediato
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Projeto: {PROJECT}")

# caminhos
SRC  = os.path.join(PROJECT, "src")
DAD  = os.path.join(PROJECT, "dados")
FIG  = os.path.join(PROJECT, "figuras")
PY   = _find_working_python()

def run_script(name, desc):
    """Executa um script .py dentro de src/ e reporta o resultado."""
    path = os.path.join(SRC, name)
    print("\n" + "=" * 70)
    print(f"STEP: {desc}")
    print(f"     {path}")
    print("=" * 70)
    if not os.path.exists(path):
        print(f"  ERRO: arquivo não encontrado: {path}")
        return False
    r = subprocess.run([PY, path], cwd=SRC,
                       capture_output=True, text=True)
    # imprime apenas a última parte do output (os resultados mais importantes)
    lines = r.stdout.strip().splitlines()
    if lines:
        print("\n".join(lines[-25:]))
    if r.returncode != 0:
        print(f"  FALHA no script {name} (exit {r.returncode})")
        print("  STDERR:\n", r.stderr[-2000:])
        return False
    print(f"  OK -> {name}")
    return True

def main():
    print("#" * 70)
    print("#  RECUPERAÇÃO DE QUEIMADAS — PIPELINE DE CLASSIFICAÇÃO")
    print("#  Projeto: recuo de flona, baseado em trabalho_kmeans")
    print("#  Uso: python3 src/run_all.py")
    print("#" * 70)

    steps = [
        ("read_index.py",  "1/4  Extração de DNs -> reflexão -> índices -> pixels.csv + histograma"),
        ("classifier.py",  "2/4  Seleção de K (silhouette x fator² / inércia), classificação, mapa poliglinal, relatório .txt"),
        ("make_histograms.py", "3/4  Histogramas adicionais de todos os índices"),
        ("gen_greyscale.py", "4/4  Imagens grayscale de cada indicador (greyscale_*.png)"),
    ]

    results = []
    for name, desc in steps:
        results.append((name, run_script(name, desc)))

    # resumo final
    print("\n" + "=" * 70)
    print("RESUMO DO PIPELINE")
    print("=" * 70)
    ok = 0
    for name, success in results:
        status = "OK  " if success else "FALHOU"
        print(f"  [{status}] {name}")
        if success:
            ok += 1
    print(f"\n  {ok}/{len(results)} scripts executados com sucesso.")
    if ok == len(results):
        print("  TODOS OS PASSOS FINALIZADOS COM SUCESSO.")
    else:
        print("  ALGUNS PASSOS FALHARAM — verifique os erros acima.")

    print(f"\n  Artefatos principais:")
    print(f"    - dados/pixels.csv")
    print(f"    - figuras/mapa_recuperacao.png")
    print(f"    - figuras/curva_k_selecao.png")
    print(f"    - figuras/indices_histograma.png")
    print(f"    - figuras/greyscale_*.png  (27 imagens grayscale, um por indicador)")
    print(f"    - recuperacao_flona.txt")
    return 0 if ok == len(results) else 1

if __name__ == "__main__": 
    sys.exit(main())
