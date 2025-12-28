import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional

#%%--------------------------------------------------------------------------------------------------
import numpy as np

def kcde1(
    train_data: dict,
    bw: str = "normal_reference",
):
    """
    KCDE via statsmodels KDEMultivariateConditional.

    IMPORTANT:
    - train_data['A'] can be:
        * (n,2) with columns (1-A, A), OR
        * (n,) scalar A in {0,1} (will be converted to (1-A, A)).
    - Output predictors read condition points from test_data.
    """
    try:
        from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional
    except Exception as e:
        raise ImportError("Please install statsmodels: pip install statsmodels") from e

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
        A2 = _get_A2(data, idx=idx)                  # (m,2)
        X = np.asarray(data["X"], dtype=float)
        if idx is not None:
            X = X[idx, :]
        return np.concatenate([A2, X], axis=1)       # (m, 2+d)

    # ---- Fit ----
    exog_train = _stack_U(train_data)               # (n, 2+d)
    endog_train = np.asarray(train_data["R"], dtype=float).reshape(-1, 1)

    d_total = exog_train.shape[1]
    # first two are discrete (one-hot), rest continuous
    indep_type = "uu" + "c" * (d_total - 2)

    mdl = KDEMultivariateConditional(
        endog=endog_train,
        exog=exog_train,
        dep_type="c",
        indep_type=indep_type,
        bw=bw,
    )

    def pdf_grid_from_test(r_grid: np.ndarray, test_data: dict, idx=None) -> np.ndarray:
        r_grid = np.asarray(r_grid, dtype=float).reshape(-1)
        U = _stack_U(test_data, idx=idx)             # (m,2+d)

        m = U.shape[0]
        G = r_grid.shape[0]
        out = np.empty((m, G), dtype=float)

        endog_pred = r_grid.reshape(-1, 1)           # (G,1)
        for i in range(m):
            ex = np.repeat(U[i:i+1, :], repeats=G, axis=0)
            out[i, :] = mdl.pdf(endog_predict=endog_pred, exog_predict=ex).reshape(-1)
        return out

    def logpdf_from_test(test_data: dict, eps: float = 1e-12, idx=None) -> np.ndarray:
        U = _stack_U(test_data, idx=idx)
        r = np.asarray(test_data["R"], dtype=float).reshape(-1)
        if idx is not None:
            r = r[idx]
        r = r.reshape(-1, 1)

        m = r.shape[0]
        out = np.empty(m, dtype=float)

        for i in range(m):
            val = mdl.pdf(endog_predict=r[i:i+1], exog_predict=U[i:i+1, :])

            # val 可能是 float / 0-d array / 1-d array（statsmodels 不同版本/输入会变）
            arr = np.asarray(val)

            if arr.ndim == 0:
                p = float(arr)                 # scalar
            else:
                p = float(arr.reshape(-1)[0])  # take first element

            out[i] = np.log(max(p, eps))

        return out

    return {
        "model": mdl,
        "pdf_grid_from_test": pdf_grid_from_test,
        "logpdf_from_test": logpdf_from_test,
    }






if __name__ == '__main__': 
    from seed import set_seed
    from data_generator import gendata_Deep
    set_seed(1145)
    train_data = gendata_Deep(n = 400, corr = 0.5)
    set_seed(4514)
    test_data = gendata_Deep(n = 200, corr = 0.5)

    kcde = kcde1(train_data, bw="normal_reference")
    
    # 选 test 中的全部条件点输出 pdf 曲线
    r_grid = np.linspace(0.1, 5.0, 100)
    pdf_all = kcde["pdf_grid_from_test"](r_grid, test_data)    # (n_test, n_grid)

    # 或者只抽 m_eval 个点（这就是你之前问的 m）
    # r_grid = np.linspace(0.1, 5.0, 400)
    # rng = np.random.default_rng(0)
    # idx = rng.choice(len(test_data["R"]), size=200, replace=False)
    # pdf_200 = kcde["pdf_grid_from_test"](r_grid, test_data, idx=idx)    # (n_test, n_grid)

    # NLL 用点密度（每个 test 样本的自身 r）
    logp = kcde["logpdf_from_test"](test_data) # (n_test, )
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
    print("KCDE Integrated L2 error (ISE):", ise)
