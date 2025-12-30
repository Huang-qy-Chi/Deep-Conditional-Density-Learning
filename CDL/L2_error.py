import numpy as np

def integrated_l2(f_hat: np.ndarray, f_true: np.ndarray, r_grid: np.ndarray) -> float:
    """
    f_hat, f_true: shape (m, G)
    r_grid: shape (G,)
    return mean_i ∫ (f_hat_i(r)-f_true_i(r))^2 dr
    """
    r_grid = np.asarray(r_grid).reshape(-1)
    diff2 = (f_hat - f_true) ** 2
    ints = np.trapezoid(diff2, r_grid, axis=1)  # (m,)
    return float(np.mean(ints))

def normalize_pdf(pdf: np.ndarray, r_grid: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """row-wise normalize to integrate to 1 (useful for CINDES MC-normalized but still numerical drift)."""
    pdf = np.maximum(pdf, 0.0)
    Z = np.trapezoid(pdf, r_grid, axis=1)
    Z = Z[:, None] 
    return pdf / np.maximum(Z, eps)
