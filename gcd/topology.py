"""
Paper III & V §8.4 — Dimension Floor Check
Implements the embedding lower bound emb(Q) lookup.
Based on Paper III: Whitney, Haefliger, and specific quotient tables.
"""
import warnings


def embedding_floor(quotient: str):
    """
    Returns minimal Euclidean dimension m* such that Q ↪ R^{m*} is possible.
    Necessary condition (Paper III, Cor 7.5). Not sufficient for learnability.
    Returns None for unknown quotients — do not pass to check_architecture.
    """
    table = {
        # Circles and spheres
        'S1': 2,           # circle; Whitney tight
        'circle': 2,
        'S2': 3,           # 2-sphere
        'S3': 4,           # 3-sphere
        # Tori — product formula emb(T^n) = n+1 (Paper III §9.2)
        'torus_T2': 3,     # T² = S¹×S¹; double pendulum config space
        'S1xS1': 3,
        'torus_T3': 4,     # T³ = (S¹)³
        # Projective spaces
        'RP2': 4,          # real projective plane; non-orientable, cannot embed in R³
        'RP3': 5,          # RP³ ≅ SO(3); Hopf 1940
        # Lie group quotients
        'SO3': 5,          # SO(3) ≅ RP³; Wall 1965
        'Klein_bottle': 4, # non-orientable surface
        # Hopf fibration
        'hopf_base': 3,    # Q = S² from Hopf S³ → S², Paper V §3.3
    }
    if quotient not in table:
        warnings.warn(
            f"Quotient '{quotient}' not in embedding floor table. "
            f"Compute emb(Q) from Paper III before training.",
            stacklevel=2
        )
        return None
    return table[quotient]


def syc_floor(d_box: float) -> int:
    """
    Sauer-Yorke-Casdagli (1991) prevalence floor for fractal compact sets.

    If m > 2*d_box, then almost every (prevalent) Lipschitz map R^k -> R^m
    is injective on A. Returns the smallest integer m satisfying this.

    IMPORTANT: This is a PREVALENCE SUFFICIENCY result, not a deterministic
    obstruction. Paper III's floor theorems give deterministic lower bounds for
    smooth manifolds. SYC gives probabilistic sufficiency for fractal sets.
    Whether a deterministic lower bound holds for fractals is an open problem
    (Paper III Open Problem 4a / Paper VI §5).

    Parameters
    ----------
    d_box : float
        Box-counting (fractal) dimension of the compact set.

    Returns
    -------
    int : smallest m such that m > 2*d_box.
    """
    import math
    return math.floor(2 * d_box) + 1


def check_architecture(latent_dim: int, quotient: str):
    """
    Returns (ok, m*) where ok = (latent_dim >= m*).
    Raises ValueError for unknown quotients.
    """
    m_star = embedding_floor(quotient)
    if m_star is None:
        raise ValueError(
            f"Cannot check architecture: embedding floor for '{quotient}' unknown. "
            f"Compute emb(Q) from Paper III before training."
        )
    return latent_dim >= m_star, m_star
