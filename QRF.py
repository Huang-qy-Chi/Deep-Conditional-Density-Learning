import numpy as np
from quantile_forest import RandomForestQuantileRegressor


#%%--------------------------------------------------------------------------------------------------
def qrf1(
    train_data: dict,
    n_estimators: int = 300,
    min_samples_leaf: int = 10,
    random_state: int = 0,
    n_taus: int = 201,
):
    try:
        from quantile_forest import RandomForestQuantileRegressor
    except Exception as e:
        raise ImportError("Please install quantile-forest: pip install quantile-forest") from e

    taus = np.linspace(0.001, 0.999, int(n_taus))

    def _get_A2(data: dict, idx=None) -> np.ndarray:
        A = np.asarray(data["A"])
        if idx is not None:
            A = A[idx]
        if A.ndim == 1:
            A0 = (1.0 - A.astype(float)).reshape(-1, 1)
            A1 = A.astype(float).reshape(-1, 1)
            return np.concatenate([A0, A1], axis=1)
        if A.ndim == 2 and A.shape[1] == 2:
            return A.astype(float)
        raise ValueError(f"Expected A shape (n,2) or (n,), got {A.shape}")

    def _stack_U(data: dict, idx=None) -> np.ndarray:
        A2 = _get_A2(data, idx=idx)
        X = np.asarray(data["X"], dtype=float)
        if idx is not None:
            X = X[idx, :]
        return np.concatenate([A2, X], axis=1)

    # ---- fit ----
    U_train = _stack_U(train_data)
    y_train = np.asarray(train_data["R"], dtype=float).reshape(-1)

    mdl = RandomForestQuantileRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(random_state),
        n_jobs=1,
    )
    mdl.fit(U_train, y_train)

    def _predict_quantiles(U: np.ndarray) -> np.ndarray:
        # quantile_forest 的 predict 往往期望 Python list，而不是 np.array
        taus_list = [float(t) for t in taus]  # <-- 关键修复

        try:
            q = mdl.predict(U, quantiles=taus_list)
        except TypeError:
            try:
                q = mdl.predict(U, q=taus_list)
            except TypeError:
                q = mdl.predict(U, alpha=taus_list)

        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            q = q.reshape(-1, 1)
        return q

    def _cdf_from_quantiles(r_grid: np.ndarray, q_mat: np.ndarray) -> np.ndarray:
        r_grid = np.asarray(r_grid, dtype=float).reshape(-1)
        m, K = q_mat.shape
        G = r_grid.shape[0]
        F = np.empty((m, G), dtype=float)
        for i in range(m):
            qi = np.maximum.accumulate(q_mat[i, :])
            F[i, :] = np.interp(r_grid, qi, taus, left=0.0, right=1.0)
        return F

    def cdf_grid_from_test(r_grid: np.ndarray, test_data: dict, idx=None) -> np.ndarray:
        r_grid = np.asarray(r_grid, dtype=float).reshape(-1)
        U = _stack_U(test_data, idx=idx)
        q = _predict_quantiles(U)
        return _cdf_from_quantiles(r_grid, q)

    def pdf_grid_from_test(r_grid: np.ndarray, test_data: dict, idx=None) -> np.ndarray:
        r_grid = np.asarray(r_grid, dtype=float).reshape(-1)
        F = cdf_grid_from_test(r_grid, test_data, idx=idx)
        pdf = np.gradient(F, r_grid, axis=1)
        return np.maximum(pdf, 0.0)

    def logpdf_from_test(test_data: dict, eps: float = 1e-12, idx=None) -> np.ndarray:
        """
        Fast pointwise density approximation using local slope of inverse CDF:
            f(r) ≈ Δtau / Δq   where q are predicted quantiles around r.
        """
        r = np.asarray(test_data["R"], dtype=float).reshape(-1)
        if idx is not None:
            r = r[idx]

        U = _stack_U(test_data, idx=idx)
        qmat = _predict_quantiles(U)  # (m,K)

        m, K = qmat.shape
        out = np.empty(m, dtype=float)

        for i in range(m):
            q = np.maximum.accumulate(qmat[i, :])
            ri = r[i]
            j = int(np.searchsorted(q, ri, side="right") - 1)

            if j < 0 or j >= K - 1:
                out[i] = np.log(eps)
                continue

            dq = q[j + 1] - q[j]
            dt = taus[j + 1] - taus[j]
            p = (dt / dq) if dq > 0 else eps
            out[i] = np.log(max(p, eps))

        return out

    return {
        "model": mdl,
        "cdf_grid_from_test": cdf_grid_from_test,
        "pdf_grid_from_test": pdf_grid_from_test,
        "logpdf_from_test": logpdf_from_test,
    }









if __name__ == '__main__': 
    from seed import set_seed
    from data_generator import gendata_Deep
    set_seed(1145)
    train_data = gendata_Deep(n = 500, corr = 0.5)
    set_seed(4514)
    test_data = gendata_Deep(n = 200, corr = 0.5)

    qrf = qrf1(train_data, n_estimators=300, min_samples_leaf=10, random_state=0)

    # 输出：测试集全部条件点的密度曲线
    r_grid = np.linspace(0.1, 5.0, 100)
    pdf_all = qrf["pdf_grid_from_test"](r_grid, test_data)  # (n_test, 500)

    # 输出：测试集抽 m_eval 个条件点的密度曲线
    # rng = np.random.default_rng(0)
    # idx = rng.choice(len(test_data["R"]), size=200, replace=False)
    # pdf_200 = qrf["pdf_grid_from_test"](r_grid, test_data, idx=idx)        # (200, 500)

    # 点密度（用于 NLL）
    logp = qrf["logpdf_from_test"](test_data)
    nll = -float(np.mean(logp))
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
    print("QRF Integrated L2 error (ISE):", ise)