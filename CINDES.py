import numpy as np

def cindes(
    train_data: dict,
    y_key: str = "R",
    a_key: str = "A",
    x_key: str = "X",
    *,
    hidden_sizes=(64, 64, 64),
    lr=1e-3,
    weight_decay=0.0,
    batch_size=256,
    epochs=200,
    device=None,
    seed=0,
    # fake Y support: by default use train min/max per dim
    y_low=None,
    y_high=None,
    # normalization (Remark 4): Monte Carlo integration with k samples
    normalize=True,
    k_norm=1024,
    eps=1e-12,
):
    """
    CINDES explicit density estimation (Algorithm 1): reduce density estimation to classification.

    Input dict format (like your simulation):
      train_data["A"] : shape (n,2) with columns (1-A, A) OR (n,1) binary A
      train_data["X"] : shape (n,dx)
      train_data["R"] : shape (n,) or (n,dy)

    Returns a KCDE-like wrapper dict with:
      - "logpdf_from_test"(test_data) -> (n_test,)
      - "pdf_grid_from_test"(r_grid, test_data, idx=None) -> (m, G) [univariate R only]
      - "pdf_point"(r, a_x) -> density at points (vectorized)
      - "model": the torch module (if torch installed)
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as e:
        raise ImportError("This cindes() implementation requires PyTorch. Please install torch.") from e

    rng = np.random.default_rng(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------- helpers to extract U=(A,X) and Y=R ----------
    def _as_2d(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    def _get_U(data: dict):
        A = _as_2d(data[a_key]).astype(np.float32)
        X = _as_2d(data[x_key]).astype(np.float32)
        return np.concatenate([A, X], axis=1)  # (n, 2+dx)

    def _get_Y(data: dict):
        Y = _as_2d(data[y_key]).astype(np.float32)  # (n, dy)
        return Y

    U_train = _get_U(train_data)
    Y_train = _get_Y(train_data)
    n, du = U_train.shape
    dy = Y_train.shape[1]

    # support for Uniform(Y): use train min/max if not provided
    if y_low is None:
        y_low = Y_train.min(axis=0)
    else:
        y_low = np.asarray(y_low, dtype=np.float32).reshape(dy)
    if y_high is None:
        y_high = Y_train.max(axis=0)
    else:
        y_high = np.asarray(y_high, dtype=np.float32).reshape(dy)

    # avoid degenerate ranges
    span = np.maximum(y_high - y_low, 1e-6)
    volY = float(np.prod(span))

    # --------- build dataset for classification ----------
    # real samples: (U_i, Y_i), label=1
    # fake samples: (U_i, Ytilde_i), label=0, where Ytilde_i ~ Unif([y_low,y_high])
    Y_fake = y_low + rng.random(size=(n, dy), dtype=np.float32) * span.astype(np.float32)
    Z_real = np.concatenate([U_train, Y_train], axis=1).astype(np.float32)
    Z_fake = np.concatenate([U_train, Y_fake], axis=1).astype(np.float32)

    Z = np.concatenate([Z_real, Z_fake], axis=0)  # (2n, du+dy)
    t = np.concatenate([np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32)], axis=0)  # (2n,)

    # shuffle
    perm = rng.permutation(2 * n)
    Z = Z[perm]
    t = t[perm]

    # --------- define MLP classifier f_theta(U,Y) ----------
    in_dim = du + dy
    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(torch.nn.Linear(prev, h))
        layers.append(torch.nn.ReLU())
        prev = h
    layers.append(torch.nn.Linear(prev, 1))  # logits
    net = torch.nn.Sequential(*layers).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    Z_torch = torch.from_numpy(Z).to(device)
    t_torch = torch.from_numpy(t).to(device)

    # mini-batch training
    net.train()
    for ep in range(int(epochs)):
        idx = rng.permutation(2 * n)
        for start in range(0, 2 * n, batch_size):
            b = idx[start:start + batch_size]
            xb = Z_torch[b]
            yb = t_torch[b].view(-1, 1)
            logits = net(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    net.eval()

    # --------- core scoring functions ----------
    @torch.no_grad()
    def _f_logits_np(U: np.ndarray, Y: np.ndarray) -> np.ndarray:
        U = np.asarray(U, dtype=np.float32)
        Y = _as_2d(Y).astype(np.float32)
        if U.ndim == 1:
            U = U.reshape(1, -1)
        if Y.shape[0] == 1 and U.shape[0] > 1:
            Y = np.repeat(Y, U.shape[0], axis=0)
        Z = np.concatenate([U, Y], axis=1).astype(np.float32)
        logits = net(torch.from_numpy(Z).to(device)).cpu().numpy().reshape(-1)
        return logits  # shape (m,)

    @torch.no_grad()
    def _logZ_mc(U: np.ndarray, k: int) -> np.ndarray:
        """
        Approximate log Z(U)=log ∫_Y exp(f(U,y)) dy
        using Monte Carlo on Uniform(Y):
          Z ≈ Vol(Y) * mean_j exp(f(U, Y_j))
        """
        U = np.asarray(U, dtype=np.float32)
        if U.ndim == 1:
            U = U.reshape(1, -1)
        m = U.shape[0]

        # sample k points once, reuse for all U (common random numbers)
        Yk = y_low + rng.random(size=(k, dy), dtype=np.float32) * span.astype(np.float32)  # (k,dy)

        # evaluate f(U_i, Yk_j): do it in chunks to save memory
        chunk = 512
        out = np.empty((m, k), dtype=np.float32)
        for j0 in range(0, k, chunk):
            j1 = min(k, j0 + chunk)
            Yc = Yk[j0:j1]  # (c,dy)
            # broadcast by repeating U for each y in chunk
            # build (m*c, du+dy)
            U_rep = np.repeat(U, repeats=(j1 - j0), axis=0)
            Y_rep = np.tile(Yc, (m, 1))
            logits = _f_logits_np(U_rep, Y_rep).reshape(m, (j1 - j0))
            out[:, j0:j1] = logits.astype(np.float32)

        # logmeanexp over k
        maxv = out.max(axis=1, keepdims=True)
        lse = maxv + np.log(np.mean(np.exp(out - maxv), axis=1, keepdims=True) + eps)
        logZ = np.log(volY + eps) + lse.reshape(-1)  # (m,)
        return logZ

    def logpdf_point(U: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        log \hat p(y|U) = f_hat(U,y) - logZ_hat(U)
        """
        f = _f_logits_np(U, y)  # (m,)
        if normalize:
            logZ = _logZ_mc(U, int(k_norm))
            return f - logZ
        else:
            # unnormalized "density-like" score exp(f); still return f as log-score
            return f

    def pdf_point(U: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(logpdf_point(U, y))

    # --------- API: from test_data ----------
    def logpdf_from_test(test_data: dict, idx=None) -> np.ndarray:
        U = _get_U(test_data)
        Y = _get_Y(test_data)
        if idx is not None:
            U = U[idx]
            Y = Y[idx]
        lp = logpdf_point(U, Y)
        return lp

    def pdf_grid_from_test(r_grid: np.ndarray, test_data: dict, idx=None) -> np.ndarray:
        """
        For univariate R only: return pdf curves for each test point at r_grid.
        Output shape: (m, G)
        """
        r_grid = np.asarray(r_grid, dtype=np.float32).reshape(-1)
        G = len(r_grid)

        U = _get_U(test_data)
        if idx is not None:
            U = U[idx]
        m = U.shape[0]

        if dy != 1:
            raise ValueError("pdf_grid_from_test is implemented for univariate R (dy=1) only.")

        # build (m*G, du) and (m*G,1)
        U_rep = np.repeat(U, repeats=G, axis=0)
        Y_rep = np.tile(r_grid.reshape(-1, 1), (m, 1))
        lp = logpdf_point(U_rep, Y_rep).reshape(m, G)
        return np.exp(lp)

    # return wrapper dict (KCDE-like)
    return {
        "name": "CINDES-explicit (Algorithm 1)",
        "model": net,
        "y_low": y_low.astype(np.float32),
        "y_high": y_high.astype(np.float32),
        "volY": volY,
        "normalize": bool(normalize),
        "k_norm": int(k_norm),
        "logpdf_point": logpdf_point,          # (U,y)->logpdf
        "pdf_point": pdf_point,                # (U,y)->pdf
        "logpdf_from_test": logpdf_from_test,  # test_data->(n_test,)
        "pdf_grid_from_test": pdf_grid_from_test,  # (r_grid,test_data)->(n_test,G) [dy=1]
    }



if __name__ == '__main__': 
    from seed import set_seed
    from data_generator import gendata_Deep
    set_seed(1145)
    train_data = gendata_Deep(n = 500, corr = 0.5)
    set_seed(4514)
    test_data = gendata_Deep(n = 200, corr = 0.5)
    # 训练
  
    cin = cindes(train_data, epochs=200, hidden_sizes=(64,64,64), normalize=True, k_norm=1024)

    

    # 输出测试集中每个条件点的密度曲线（dy=1）
    r_grid = np.linspace(0.1, 5.0, 100).astype(np.float32)
    pdf_all = cin["pdf_grid_from_test"](r_grid, test_data)  # (n_test, 100)

    # 点密度（用于 NLL）
    logp = cin["logpdf_from_test"](test_data)      # shape (n_test,)
    nll  = -float(np.mean(logp))


    # 输出：测试集抽 m_eval 个条件点的密度曲线
    # rng = np.random.default_rng(0)
    # idx = rng.choice(len(test_data["R"]), size=200, replace=False)
    # pdf_200 = flex["pdf_grid_from_test"](r_grid, test_data, idx=idx)       # (200, 500)

    # 点密度（用于 NLL）
    print("NLL:", nll)
    print(pdf_all.shape)
    print(type(pdf_all))
    print(logp.shape)
    print(type(logp))

    from L2_error import integrated_l2
    from L2_error import normalize_pdf
    from true_pdf import true_pdf_grid_fn
    pdf_all = np.asarray(pdf_all, dtype=float)
    A_test = np.asarray(test_data["A"])
    X_test = np.asarray(test_data["X"], dtype=float)
    if A_test.ndim == 2 and A_test.shape[1] == 2:
        A_scalar = A_test[:, 1].astype(int)
    else:
        A_scalar = A_test.astype(int)

    X_eval = X_test

    f_hat = normalize_pdf(pdf_all, r_grid)
    f_true = true_pdf_grid_fn(r_grid, A_scalar, X_eval)
    f_true = np.asarray(f_true, dtype=float)
    f_true = normalize_pdf(f_true, r_grid)

    ise = integrated_l2(f_hat, f_true, r_grid)
    print("CINDES Integrated L2 error (ISE):", ise)
