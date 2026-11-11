import numpy

def kmeans(x, y, K=2, max_iter=300, tol=1e-6, seed=0):
    """
    rotulo : ndarray (N,) — rótulo de cluster de cada ponto
    cx, cy : ndarray (K,) — coordenadas finais dos centros
    n_iter : int — número de iterações executadas
    J_final : float — inércia final (soma das dist^2 a seu próprio centro)
    """
    rng = numpy.random.default_rng(seed)
    N = x.shape[0]
    # 1: escolher K pontos reais como sementes
    idx = rng.choice(N, size=K, replace=False)
    cx = x[idx].copy()
    cy = y[idx].copy()

    rotulo = numpy.zeros(N, dtype=int)
    J_final = None
    for it in range(1, max_iter + 1):
        # 2: matriz distancia (N, K)
        D = numpy.empty((N, K))
        for k in range(K):
            D[:, k] = numpy.sqrt((x - cx[k]) ** 2 + (y - cy[k]) ** 2)
        rotulo = numpy.argmin(D, axis=1)

        # 3: recalcular centros
        cx_novo = cx.copy()
        cy_novo = cy.copy()
        for k in range(K):
            memb = rotulo == k
            if numpy.any(memb):
                cx_novo[k] = x[memb].mean()
                cy_novo[k] = y[memb].mean()

        # 4: convergência (deslocamento dos centros)
        desloc = numpy.sqrt((cx_novo - cx) ** 2 + (cy_novo - cy) ** 2)
        cx, cy = cx_novo, cy_novo

        # 5: inércia
        J = float((D[numpy.arange(N), rotulo] ** 2).sum())
        J_final = J
        if desloc.max() < tol:
            break
    return rotulo, cx, cy, it, J_final


def kmeans_nd(X, K=2, max_iter=300, tol=1e-4, seed=42):
    """
    K-means clássico em dimensão N.
    Retorna (rotulo, centros, n_iter, J_final)
    """

    N = X.shape[0]
    rng = numpy.random.default_rng(seed)
    idx = rng.choice(N, size=K, replace=False)
    centros = X[idx].copy()

    rotulo = numpy.zeros(N, dtype=int)
    J_final = None
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        # distância euclidiana (N, K)
        D = numpy.empty((N, K))
        for k in range(K):
            diff = X - centros[k]
            D[:, k] = numpy.sqrt((diff ** 2).sum(axis=1))

        rotulo = numpy.argmin(D, axis=1)

        centros_novo = centros.copy()
        for k in range(K):
            memb = rotulo == k
            if numpy.any(memb):
                centros_novo[k] = X[memb].mean(axis=0)

        desloc = numpy.sqrt(((centros_novo - centros) ** 2).sum(axis=1)).max()
        centros = centros_novo

        J = float((D[numpy.arange(N), rotulo] ** 2).sum())
        J_final = J
        if desloc < tol:
            break
    return rotulo, centros, n_iter, J_final