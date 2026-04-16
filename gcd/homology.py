"""
Topology certification via persistent homology.
Paper VI §4 — Circularity and torus scores using ripser.

These are the ground-truth topology metrics used in all Paper VI experiments.
They are NOT simulated — they compute actual persistent homology on the
latent point cloud.
"""
import numpy as np

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


def _check_ripser():
    if not RIPSER_AVAILABLE:
        raise ImportError(
            "ripser is required for topology certification. "
            "Install with: pip install ripser"
        )


def circularity_score(z: np.ndarray, threshold: float = 0.0) -> float:
    """
    Paper VI §4: Circularity score for S¹ recovery.

    Defined as the lifetime of the longest H₁ persistence bar in the
    Vietoris–Rips filtration on latent point cloud z.

    For a point cloud on S¹: score is large (bounded away from 0).
    For a collapsed line (traversal trap): score ≈ 0.

    Parameters
    ----------
    z : np.ndarray, shape [N, d]
        Latent point cloud.
    threshold : float
        Minimum bar lifetime to consider (noise floor). Default 0.

    Returns
    -------
    float : lifetime of longest H₁ bar. Range [0, ∞).
    """
    _check_ripser()
    # Subsample for speed if large
    if len(z) > 500:
        idx = np.random.choice(len(z), 500, replace=False)
        z = z[idx]
    # Normalise to unit scale for consistent thresholds
    z = z - z.mean(axis=0)
    scale = np.linalg.norm(z, axis=1).mean()
    if scale > 0:
        z = z / scale
    dgms = ripser(z, maxdim=1)['dgms']
    h1 = dgms[1]  # H₁ persistence diagram
    if len(h1) == 0:
        return 0.0
    lifetimes = h1[:, 1] - h1[:, 0]
    # Filter out infinite bars
    finite = lifetimes[np.isfinite(lifetimes)]
    if len(finite) == 0:
        return 0.0
    return float(finite.max())


def torus_score(z: np.ndarray, threshold: float = 0.3) -> dict:
    """
    Paper VI §4: Torus score for T² recovery.

    T² = S¹ × S¹ has Betti numbers β₀=1, β₁=2, β₂=1.
    We certify T² by requiring two independent persistent H₁ classes.

    Parameters
    ----------
    z : np.ndarray, shape [N, d]
        Latent point cloud.
    threshold : float
        Minimum bar lifetime for a class to count as persistent. Default 0.3.

    Returns
    -------
    dict with keys:
        'betti_1'   : int, number of persistent H₁ classes above threshold
        'certified' : bool, True if betti_1 >= 2 (T² topology)
        'lifetimes' : list of float, sorted lifetimes of persistent H₁ bars
    """
    _check_ripser()
    if len(z) > 500:
        idx = np.random.choice(len(z), 500, replace=False)
        z = z[idx]
    z = z - z.mean(axis=0)
    scale = np.linalg.norm(z, axis=1).mean()
    if scale > 0:
        z = z / scale
    dgms = ripser(z, maxdim=1)['dgms']
    h1 = dgms[1]
    if len(h1) == 0:
        return {'betti_1': 0, 'certified': False, 'lifetimes': []}
    lifetimes = h1[:, 1] - h1[:, 0]
    finite = lifetimes[np.isfinite(lifetimes)]
    persistent = sorted([float(l) for l in finite if l > threshold], reverse=True)
    betti_1 = len(persistent)
    return {
        'betti_1': betti_1,
        'certified': betti_1 >= 2,
        'lifetimes': persistent
    }
