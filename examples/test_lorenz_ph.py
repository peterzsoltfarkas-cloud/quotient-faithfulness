"""
Empirical exploration: Persistent Homology of the Lorenz Attractor
Paper VI — beyond smooth manifolds (open problem)

This script computes actual persistent homology (via ripser) on the Lorenz
attractor and reports what is found. It does NOT claim to validate any
theorem from Papers I–V. The Lorenz attractor is a fractal strange attractor
in R³ with Hausdorff dimension ≈ 2.06 and topological covering dimension 2.
Paper III's dimension floor theorems apply to compact smooth manifolds and
compact metric spaces; extending them to fractal attractors is an open problem
stated in Paper VI §5.

What this script does:
  1. Generate a Lorenz trajectory via RK45 integration
  2. Subsample to a manageable point cloud
  3. Compute H0, H1, H2 persistent homology at projection dimensions m=2..6
  4. Report the results honestly — whatever ripser finds

Ripser is required: pip install ripser
If ripser is not installed the script reports that and exits cleanly.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False

PERSISTENCE_THRESHOLD = 0.15  # bars shorter than this are treated as noise


def lorenz(t, state, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]


def generate_lorenz(n_points=800):
    """Integrate Lorenz system and return a clean point cloud."""
    sol = solve_ivp(
        lorenz, [0, 100], [1.0, 1.0, 1.0],
        t_eval=np.linspace(0, 100, 20000),
        method='RK45', rtol=1e-9
    )
    raw = sol.y.T[5000:]           # discard transient
    step = max(1, len(raw) // n_points)
    pts = raw[::step][:n_points]
    # Normalise to unit scale for consistent filtration threshold
    pts = pts - pts.mean(axis=0)
    pts = pts / np.linalg.norm(pts, axis=1).mean()
    return pts


def count_persistent(dgm, threshold=PERSISTENCE_THRESHOLD):
    """Count bars with lifetime > threshold (excluding infinite bars)."""
    if len(dgm) == 0:
        return 0
    finite = dgm[dgm[:, 1] < np.inf]
    return int(np.sum(finite[:, 1] - finite[:, 0] > threshold))


def run_ph(points, m, maxdim=2):
    """Project to R^m and compute persistent homology."""
    proj = points[:, :m] if m <= points.shape[1] else \
        np.hstack([points, np.zeros((len(points), m - points.shape[1]))])
    dgms = ripser(proj, maxdim=maxdim)['dgms']
    return {
        'H0': count_persistent(dgms[0]),
        'H1': count_persistent(dgms[1]),
        'H2': count_persistent(dgms[2]) if len(dgms) > 2 else 0,
        'dgms': dgms
    }


if __name__ == '__main__':
    print("=" * 70)
    print("Lorenz Attractor — Persistent Homology (empirical, Paper VI §5)")
    print("=" * 70)

    if not RIPSER_AVAILABLE:
        print("\nripser is not installed.")
        print("Install with:  pip install ripser")
        print("This script requires real computation and cannot simulate results.")
        exit(1)

    print("\nGenerating Lorenz trajectory (RK45, discarding transient)...")
    points = generate_lorenz(n_points=800)
    print(f"Point cloud: {len(points)} points in R³, normalised to unit scale")
    print(f"Kaplan-Yorke dimension ≈ 2.06 (Lorenz 1963, standard parameters)")

    # SYC floor
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from gcd.topology import syc_floor
    d_box = 2.062
    m_syc = syc_floor(d_box)
    print(f"SYC floor (Sauer-Yorke-Casdagli 1991): m > 2*{d_box} = {2*d_box:.3f}")
    print(f"  -> smallest integer m satisfying this: m = {m_syc}")
    print(f"  -> prevalence result (not deterministic lower bound)")
    print(f"Topological covering dimension: 2")
    print(f"Note: Paper III's theorems apply to compact smooth manifolds.")
    print(f"      This is an empirical exploration outside that scope.")
    print()

    results = {}
    for m in range(2, 7):
        print(f"  Computing H0, H1, H2 at m={m}...", end=' ', flush=True)
        results[m] = run_ph(points, m)
        print(f"H0={results[m]['H0']}  H1={results[m]['H1']}  H2={results[m]['H2']}")

    print()
    print(f"Persistence threshold: {PERSISTENCE_THRESHOLD}")
    print()
    print("Persistent Betti numbers (actual ripser computation):")
    print("m | H0 | H1 | H2 | Note")
    print("--|----|----|----|---------------------------------")
    for m in range(2, 7):
        r = results[m]
        note = "(original R³)" if m == 3 else ""
        print(f"{m} | {r['H0']:2d} | {r['H1']:2d} | {r['H2']:2d} | {note}")

    print()
    print("Interpretation:")
    print("  These are empirical observations, not theoretical predictions.")
    print("  A change in H1 or H2 between projection dimensions indicates")
    print("  topological information is being lost at that dimension.")
    print("  Whether this constitutes a 'dimension floor' for the Lorenz")
    print("  attractor in the sense of Paper III is an open question.")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(
        'Lorenz Attractor: Persistent Homology (Empirical)\n'
        'Note: outside scope of Paper III smooth-manifold theorems — Paper VI §5',
        fontsize=11, fontweight='bold'
    )

    # Left: attractor in R³
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(points[:, 0], points[:, 1], points[:, 2],
             'b-', linewidth=0.4, alpha=0.3)
    c = np.linspace(0, 1, len(points))
    ax1.scatter(points[::5, 0], points[::5, 1], points[::5, 2],
                c=c[::5], cmap='plasma', s=8, alpha=0.6)
    ax1.set_title('Lorenz attractor\n(dim ≈ 2.06, fractal)', fontsize=9)
    ax1.set_xlabel('x', fontsize=8)
    ax1.set_ylabel('y', fontsize=8)
    ax1.set_zlabel('z', fontsize=8)
    ax1.view_init(elev=25, azim=45)

    # Middle: Betti numbers vs m
    ax2 = fig.add_subplot(132)
    ms = list(range(2, 7))
    ax2.plot(ms, [results[m]['H0'] for m in ms], 'bo-',
             linewidth=2, markersize=8, label='H₀')
    ax2.plot(ms, [results[m]['H1'] for m in ms], 'ro-',
             linewidth=2, markersize=8, label='H₁ (loops)')
    ax2.plot(ms, [results[m]['H2'] for m in ms], 'go-',
             linewidth=2, markersize=8, label='H₂ (voids)')
    ax2.set_xlabel('Projection dimension m', fontsize=10)
    ax2.set_ylabel('Persistent Betti number', fontsize=10)
    ax2.set_title('Betti numbers vs projection dimension\n(actual ripser results)',
                  fontsize=9)
    ax2.set_xticks(ms)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Right: persistence diagram at m=3 (original embedding)
    ax3 = fig.add_subplot(133)
    dgms = results[3]['dgms']
    colours = ['#2196F3', '#F44336', '#4CAF50']
    for i, (dgm, col) in enumerate(zip(dgms[:3], colours)):
        if len(dgm) > 0:
            finite = dgm[dgm[:, 1] < np.inf]
            if len(finite) > 0:
                ax3.scatter(finite[:, 0], finite[:, 1],
                            s=40, c=col, alpha=0.7, label=f'H{i}')
    lim = ax3.get_xlim()[1] if ax3.get_xlim()[1] > 0 else 2.0
    ax3.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax3.set_xlabel('Birth', fontsize=10)
    ax3.set_ylabel('Death', fontsize=10)
    ax3.set_title('Persistence diagram at m=3\n(Lorenz in original R³)',
                  fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test3_lorenz_ph.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("\nSaved test3_lorenz_ph.png")
    print("These results should be reported in Paper VI §5 as empirical")
    print("observations with appropriate caveats about scope.")
