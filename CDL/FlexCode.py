import numpy as np
from flexcode import FlexCodeModel
from flexcode.regression_models import NN, RandomForest, XGBoost, Lasso  # 指定 NN 模型

#%%-----------------------------------------------------------------------------------------------
def flexcode1(
    train_data: dict,
    z_min: float,
    z_max: float,
    max_basis: int = 30,
    regressor_factory=None,
    random_state=0
):
    """
    FlexCode wrapper (dict-in, dict-out) robust to flexcode.predict() return formats.

    - Supports A as (n,2) one-hot (1-A, A) or scalar (n,)
    - pdf_grid_from_test reads condition points from test_data
    - Handles flexcode.predict possibly returning (grid, pdf) or list-of-arrays
    """
    try:
        from flexcode import FlexCodeModel
    except Exception as e:
        raise ImportError("Please install flexcode: pip install flexcode") from e

    # default regressor factory
    # if regressor_factory is None:
    #     from sklearn.ensemble import RandomForestRegressor
    #     def regressor_factory(max_basis, regression_params, custom_model):
    #         return RandomForestRegressor(
    #             n_estimators=200,
    #             min_samples_leaf=5,
    #             random_state=random_state,
    #             n_jobs=1,
    #         )

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

    def _as_pdf_matrix(pred, m: int, G: int) -> np.ndarray:
        """
        Convert flexcode.predict output to a numeric array of shape (m, G).
        Handles:
          - pred is ndarray (m,G) or (G,) when m=1
          - pred is (grid, pdf) tuple/list
          - pred is list of length m with each element length G
        """
        # Case 1: tuple/list like (grid, pdf)
        if isinstance(pred, (tuple, list)) and len(pred) == 2:
            # try to detect which element is pdf
            a0 = np.asarray(pred[0], dtype=object)
            a1 = np.asarray(pred[1], dtype=object)

            # prefer the element that can become a numeric matrix
            # usually pred[1] is pdf
            candidate = pred[1]
        else:
            candidate = pred

        # Case 2: list-of-arrays (length m)
        if isinstance(candidate, list):
            # try stack
            try:
                mat = np.vstack([np.asarray(row, dtype=float).reshape(1, -1) for row in candidate])
            except Exception as e:
                raise ValueError("flexcode.predict returned a list that cannot be stacked into (m,G).") from e
        else:
            # Case 3: already array-like
            mat = np.asarray(candidate)

            # Sometimes still object array; try force numeric
            if mat.dtype == object:
                # attempt vstack if it's array of sequences
                try:
                    mat = np.vstack([np.asarray(row, dtype=float).reshape(1, -1) for row in mat])
                except Exception as e:
                    raise ValueError("flexcode.predict returned an object array that cannot be coerced to float matrix.") from e
            else:
                mat = mat.astype(float, copy=False)

        # Normalize shape
        if mat.ndim == 1:
            # (G,) -> (1,G)
            mat = mat.reshape(1, -1)

        if mat.shape[0] != m or mat.shape[1] != G:
            # some versions return transposed?
            if mat.shape[0] == G and mat.shape[1] == m:
                mat = mat.T
            else:
                raise ValueError(f"Unexpected pdf matrix shape {mat.shape}, expected ({m},{G}).")
        return mat

    # ---- fit ----
    U_train = _stack_U(train_data)
    z_train = np.asarray(train_data["R"], dtype=float).reshape(-1)

    mdl = FlexCodeModel(
        # model=regressor_factory,          # MUST be callable, not string
        model=NN,
        max_basis=int(max_basis),
        basis_system="cosine",
        z_min=float(z_min),
        z_max=float(z_max),
        regression_params={},
        # regression_params={"k": 20},  # NN 的参数，例如邻居数 k=20
        custom_model=None,
    )
    mdl.fit(x_train=U_train, z_train=z_train)

    # def pdf_grid_from_test(r_grid: np.ndarray, test_data: dict, idx=None, dense_G: int = 1500,\
    #                         eps: float = 1e-12) -> np.ndarray:
    #     r_grid = np.asarray(r_grid, dtype=float).reshape(-1)
    #     G = len(r_grid)

    #     U = _stack_U(test_data, idx=idx)   # (m, p)
    #     m = U.shape[0]

    #     out = np.empty((m, G), dtype=float)

    #     for i in range(m):
    #         pred_i = mdl.predict(x_new=U[i:i+1, :], n_grid=G)

    #         # flexcode 可能返回 (grid, pdf) 或 pdf
    #         if isinstance(pred_i, (tuple, list)) and len(pred_i) == 2:
    #             pred_i = pred_i[1]  # take pdf part

    #         arr = np.asarray(pred_i)
    #         # 常见形状：(G,1), (1,G), (G,)
    #         if arr.ndim == 2:
    #             if arr.shape == (G, 1):
    #                 arr = arr[:, 0]
    #             elif arr.shape == (1, G):
    #                 arr = arr[0, :]
    #             else:
    #                 # 兜底：拉平取前G个
    #                 arr = arr.reshape(-1)[:G]
    #         else:
    #             arr = arr.reshape(-1)[:G]

    #         out[i, :] = arr.astype(float, copy=False)
        

    #     return out

    # def pdf_grid_from_test(r_grid: np.ndarray, test_data: dict, idx=None, dense_G: int = 1500) -> np.ndarray: r_grid = np.asarray(r_grid, dtype=float).reshape(-1), G = len(r_grid)
    #     U = _stack_U(test_data, idx=idx)   # (m, p)
    #     m = U.shape[0]

    #     # 与 logpdf_from_test 一致地使用统一稠密网格进行预测
    #     dense_grid = np.linspace(float(z_min), float(z_max), int(dense_G))

    #     out = np.empty((m, G), dtype=float)

    #     for i in range(m):
    #         pred_i = mdl.predict(x_new=U[i:i+1, :], n_grid=len(dense_grid))

    #         # 兼容 (grid, pdf) 或直接 pdf
    #         if isinstance(pred_i, (tuple, list)) and len(pred_i) == 2:
    #             pred_i = pred_i[1]

    #         arr = np.asarray(pred_i)
    #         # 常见形状：(G,1), (1,G), (G,)
    #         if arr.ndim == 2:
    #             if arr.shape == (len(dense_grid), 1):
    #                 pdf_i = arr[:, 0]
    #             elif arr.shape == (1, len(dense_grid)):
    #                 pdf_i = arr[0, :]
    #             else:
    #                 pdf_i = arr.reshape(-1)[:len(dense_grid)]
    #         else:
    #             pdf_i = arr.reshape(-1)[:len(dense_grid)]

    #         # 在用户给定的 r_grid 上插值，边界外置 0
    #         out[i, :] = np.interp(r_grid, dense_grid, pdf_i, left=0.0, right=0.0)

    #     return out
    

    # def logpdf_from_test(test_data: dict, eps: float = 1e-12, idx=None, dense_G: int = 1500) -> np.ndarray:
    #     r = np.asarray(test_data["R"], dtype=float).reshape(-1)
    #     if idx is not None:
    #         r = r[idx]

    #     U = _stack_U(test_data, idx=idx)
    #     m = U.shape[0]

    #     grid = np.linspace(float(z_min), float(z_max), int(dense_G))
    #     out = np.empty(m, dtype=float)

    #     for i in range(m):
    #         pred_i = mdl.predict(x_new=U[i:i+1, :], n_grid=len(grid))
    #         if isinstance(pred_i, (tuple, list)) and len(pred_i) == 2:
    #             pred_i = pred_i[1]

    #         arr = np.asarray(pred_i)
    #         if arr.ndim == 2:
    #             if arr.shape == (len(grid), 1):
    #                 pdf_i = arr[:, 0]
    #             elif arr.shape == (1, len(grid)):
    #                 pdf_i = arr[0, :]
    #             else:
    #                 pdf_i = arr.reshape(-1)[:len(grid)]
    #         else:
    #             pdf_i = arr.reshape(-1)[:len(grid)]

    #         p = float(np.interp(r[i], grid, pdf_i, left=0.0, right=0.0))
    #         out[i] = np.log(max(p, eps))

    #     return out
    def pdf_grid_from_test(
        r_grid: np.ndarray,
        test_data: dict,
        idx=None,
        dense_G: int = 1500,
    ) -> np.ndarray:
        """
        输出大小: (n_test, len(r_grid))
        计算方式与 logpdf_from_test 完全一致：先在 dense grid 上预测，再插值到 r_grid。
        """
        r_grid = np.asarray(r_grid, dtype=float).reshape(-1)
        G = len(r_grid)

        U = _stack_U(test_data, idx=idx)
        m = U.shape[0]

        # 与 logpdf_from_test 使用同一条“预测基准网格”
        grid = np.linspace(float(z_min), float(z_max), int(dense_G))

        out = np.empty((m, G), dtype=float)

        for i in range(m):
            pred_i = mdl.predict(x_new=U[i:i+1, :], n_grid=len(grid))
            if isinstance(pred_i, (tuple, list)) and len(pred_i) == 2:
                pred_i = pred_i[1]  # take pdf part

            arr = np.asarray(pred_i)

            # 规范化成一维 pdf_i: (len(grid),)
            if arr.ndim == 2:
                if arr.shape == (len(grid), 1):
                    pdf_i = arr[:, 0]
                elif arr.shape == (1, len(grid)):
                    pdf_i = arr[0, :]
                else:
                    pdf_i = arr.reshape(-1)[:len(grid)]
            else:
                pdf_i = arr.reshape(-1)[:len(grid)]

            # 插值到用户给定 r_grid
            out[i, :] = np.interp(r_grid, grid, pdf_i, left=0.0, right=0.0)

        return out


    def logpdf_from_test(
        test_data: dict,
        eps: float = 1e-12,
        idx=None,
        dense_G: int = 1500,
    ) -> np.ndarray:
        """
        输出大小: (n_test,)
        与 pdf_grid_from_test 的计算路径一致，只是在每个样本的观测 R[i] 处取密度并取 log。
        """
        r = np.asarray(test_data["R"], dtype=float).reshape(-1)
        if idx is not None:
            r = r[idx]

        U = _stack_U(test_data, idx=idx)
        m = U.shape[0]

        grid = np.linspace(float(z_min), float(z_max), int(dense_G))
        out = np.empty(m, dtype=float)

        for i in range(m):
            pred_i = mdl.predict(x_new=U[i:i+1, :], n_grid=len(grid))
            if isinstance(pred_i, (tuple, list)) and len(pred_i) == 2:
                pred_i = pred_i[1]

            arr = np.asarray(pred_i)

            if arr.ndim == 2:
                if arr.shape == (len(grid), 1):
                    pdf_i = arr[:, 0]
                elif arr.shape == (1, len(grid)):
                    pdf_i = arr[0, :]
                else:
                    pdf_i = arr.reshape(-1)[:len(grid)]
            else:
                pdf_i = arr.reshape(-1)[:len(grid)]

            p = float(np.interp(r[i], grid, pdf_i, left=0.0, right=0.0))
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
    set_seed(3456)
    train_data = gendata_Deep(n = 500, corr = 0.5)
    set_seed(4514)
    test_data = gendata_Deep(n = 200, corr = 0.5)
    # 训练
    r_lo = 0
    r_hi = np.max(train_data['R'])



    flex = flexcode1(train_data, z_min=r_lo, z_max=r_hi, max_basis=30)

    # 输出：测试集全部条件点的密度曲线
    r_grid = np.linspace(0.1, 5.0, 100)
    pdf_all = flex["pdf_grid_from_test"](r_grid, test_data)  # (n_test, 500)

    # 输出：测试集抽 m_eval 个条件点的密度曲线
    # rng = np.random.default_rng(0)
    # idx = rng.choice(len(test_data["R"]), size=200, replace=False)
    # pdf_200 = flex["pdf_grid_from_test"](r_grid, test_data, idx=idx)       # (200, 500)

    # 点密度（用于 NLL）
    logp = flex["logpdf_from_test"](test_data)
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
    f_true = true_pdf_grid_fn(r_grid, A_scalar, X_eval, model = 'deep')
    f_true = np.asarray(f_true, dtype=float)
    f_true = normalize_pdf(f_true, r_grid)

    ise = integrated_l2(f_hat, f_true, r_grid)
    print("FlexCode Integrated L2 error (ISE):", ise)