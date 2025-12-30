import numpy as np
from math import erf, sqrt, pi

# ---- standard normal pdf/cdf without scipy ----
def _phi(x):
    return np.exp(-0.5 * x * x) / sqrt(2.0 * pi)

def _Phi(x):
    # Φ(x) = 0.5*(1+erf(x/sqrt(2)))
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))

def true_pdf_grid_fn(
    r_grid: np.ndarray,
    A_scalar: np.ndarray,
    X: np.ndarray,
    *,
    model: str = "deep",     # "linear" or "deep"
    d: int | None = None,    # if provided, use first d columns; outcome depends on first 5 anyway when d>5
    trunc_eps: float = 1.0,  # epsilon truncated to [-trunc_eps, trunc_eps]; manuscript uses 1
) -> np.ndarray:
    """
    True conditional density f(r | A, X) under manuscript simulation:
      log R = mu0(X)*(1-A) + mu1(X)*A + eps,
      eps ~ N(0,1) truncated to [-1,1],
      X truncated to [-1,1] (but conditioning on X, so only affects mu evaluation domain).

    Inputs:
      r_grid: (G,) positive grid of r values
      A_scalar: (m,) in {0,1}
      X: (m,d) covariates (will only use first 5 coords for mu when available)
      model: "linear" or "deep"
      d: optional, use X[:, :d] (for safety). outcome depends on first 5 coords per manuscript.
    Output:
      pdf: (m,G) matrix, pdf[i,g] = f(r_grid[g] | A_i, X_i)
    """
    r_grid = np.asarray(r_grid, dtype=float).reshape(-1)
    A_scalar = np.asarray(A_scalar).reshape(-1)
    X = np.asarray(X, dtype=float)
    if d is not None:
        X = X[:, :int(d)]

    m = X.shape[0]
    G = r_grid.shape[0]
    if A_scalar.shape[0] != m:
        raise ValueError(f"A_scalar length {A_scalar.shape[0]} != X rows {m}")

    # Use first 5 covariates for mu (per manuscript; for d>5, outcome does not rely on X6..X15)
    if X.shape[1] < 5:
        raise ValueError(f"X must have at least 5 columns to match manuscript mu definitions, got {X.shape[1]}")
    X1, X2, X3, X4, X5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

    # (Optional) enforce truncation domain for numerical stability (manuscript truncates X to [-1,1])
    X1 = np.clip(X1, -1.0, 1.0)
    X2 = np.clip(X2, -1.0, 1.0)
    X3 = np.clip(X3, -1.0, 1.0)
    X4 = np.clip(X4, -1.0, 1.0)
    X5 = np.clip(X5, -1.0, 1.0)

    model = model.lower()
    if model not in ("linear", "deep"):
        raise ValueError("model must be 'linear' or 'deep'")

    # ---- mu0, mu1 as in manuscript (d=5 case) ----
    if model == "linear":
        mu0 = (-0.5
               + 0.25 * X1 - 0.75 * X2 + 1.0 * X3 - 0.75 * X4 + 0.25 * X5)
        mu1 = (-0.5
               + 0.75 * X1 - 0.5 * X2 + 0.5 * X3 - 0.5 * X5)
    else:  # "deep"
        # guard: X5+2 > 0 always since X5 in [-1,1]
        logX5p2 = np.log(X5 + 2.0)
        # guard sqrt(X4+1) with clipping
        sqrtX4p1 = np.sqrt(np.maximum(X4 + 1.0, 0.0))

        common = (-0.5
                  + 0.25 * X1 * X2
                  + np.sin(X3) / 3.0
                  - np.cos(X4 * logX5p2) / 2.0)

        mu0 = (common
               + np.cos(X1**2 + 2.0 * X2**2 + X3**3 + sqrtX4p1 * logX5p2 / 20.0))

        mu1 = (common
               + np.sin(X1 / 3.0
                        + np.exp(X2) / 4.0
                        + np.cos(X3 * X4)
                        - np.log(X5 + 2.0)
                        - 0.45))

    # select mu(A,X)
    mu = np.where(A_scalar.astype(int) == 1, mu1, mu0)  # (m,)

    # ---- density of R via truncated normal on eps = logR - mu ----
    # normalization constant for eps ~ N(0,1) truncated to [-c, c]
    c = float(trunc_eps)
    Z = float(_Phi(np.array([c]))[0] - _Phi(np.array([-c]))[0])  # Φ(c)-Φ(-c)

    # handle r_grid <= 0 safely
    pdf = np.zeros((m, G), dtype=float)
    pos = r_grid > 0
    if not np.any(pos):
        return pdf

    r_pos = r_grid[pos]
    log_r = np.log(r_pos)[None, :]             # (1,Gpos)
    t = log_r - mu[:, None]                    # (m,Gpos) = eps values implied by r

    mask = (np.abs(t) <= c)
    # f_logR(z|A,X) = phi(z-mu)/Z for z in [mu-c,mu+c]
    f_log = _phi(t) / max(Z, 1e-15)
    # transform: f_R(r) = f_logR(log r) * 1/r
    f_r = f_log / r_pos[None, :]
    f_r = np.where(mask, f_r, 0.0)

    pdf[:, pos] = f_r
    return pdf





if __name__ == "__main__":
    from seed import set_seed
    from data_generator import gendata_Deep
    set_seed(1145)
    train_data = gendata_Deep(n = 500, corr = 0.5)
    set_seed(4514)
    test_data = gendata_Deep(n = 200, corr = 0.5)
    A = test_data['A']
    A_scalar = A[:,1]
    X_eval = test_data['X']
    r_grid = np.linspace(0.1, 5.0, 100).astype(np.float32)
    f_true = true_pdf_grid_fn(r_grid, A_scalar, X_eval, model="deep", d=X_eval.shape[1])
    print(f_true.shape)
    print(type(f_true))