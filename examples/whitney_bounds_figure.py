"""
Figure: Paper III Embedding Dimension Lower Bounds
Whitney, Haefliger, and specific quotients.

Generates the figure for Paper III §lookup_table.
Only includes manifolds covered by Paper III's theory (compact smooth
manifolds and compact metric spaces). Lorenz and KS attractors are NOT
included here — they are outside the scope of Paper III.

Output: whitney_bounds_paper3.png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Data: only entries covered by Paper III theory ────────────────────────────
manifolds = ['S¹', 'S²', 'T²', 'S³', 'SO(3)', 'CP²']
labels = ['Circle', 'Sphere\n(Hopf base)', 'Torus', '3-sphere',
          'Rotation\ngroup', 'Complex\nprojective']
intrinsic_dim  = [1,  2,  2,  3,  3,  4]
emb_lower      = [2,  3,  3,  4,  5,  7]   # tight lower bounds
whitney_generic = [2*n+1 for n in intrinsic_dim]  # Whitney 2n+1 for general compact metric
sources = [
    'Whitney tight',
    'Whitney tight',
    'Product formula\n(Paper III §9.2)',
    'Whitney tight',
    'Wall 1965\n(Hopf + linking)',
    'Signature obstruction\n(σ=1) +\nCappell–Shaneson 1974',
]

x = np.arange(len(manifolds))
width = 0.28

fig, ax = plt.subplots(figsize=(13, 6))

b1 = ax.bar(x - width, intrinsic_dim, width, label='Intrinsic dim $n$',
            color='#5dade2', alpha=0.85, zorder=3)
b2 = ax.bar(x, emb_lower, width, label='$\\mathrm{emb}(Q)$ (tight lower bound)',
            color='#e74c3c', alpha=0.85, zorder=3)
b3 = ax.bar(x + width, whitney_generic, width,
            label='Whitney generic $2n+1$\n(compact metric spaces)',
            color='#95a5a6', alpha=0.55, zorder=3, linestyle='--',
            edgecolor='#7f8c8d', linewidth=1)

# Annotate emb values
for i, (bar, val) in enumerate(zip(b2, emb_lower)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            str(val), ha='center', va='bottom', fontweight='bold',
            fontsize=11, color='#c0392b')

# Annotate sources
for i, src in enumerate(sources):
    ax.text(x[i], -1.2, src, ha='center', va='top', fontsize=6.5,
            color='#555', style='italic')

ax.set_xlabel('Manifold / Quotient space', fontsize=12)
ax.set_ylabel('Dimension', fontsize=12)
ax.set_title('Paper III: Embedding Dimension Lower Bounds\n'
             'Whitney, Haefliger, and Specific Quotients',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'${m}$' for m in manifolds], fontsize=12)
ax.set_ylim(-2.5, 14)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, axis='y', alpha=0.3, zorder=0)
ax.axhline(0, color='k', linewidth=0.5)

# Key observation box
ax.text(0.98, 0.97,
        'Key: emb(Q) < Whitney 2n+1 for smooth manifolds\n'
        '      (Haefliger connectivity savings)\n'
        '      emb(CP²)=7 requires signature obstruction,\n'
        '      not Stiefel–Whitney classes alone',
        transform=ax.transAxes, fontsize=8,
        va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow',
                  edgecolor='#aaa', alpha=0.9))

plt.tight_layout()
plt.savefig('whitney_bounds_paper3.png', dpi=220, bbox_inches='tight')
plt.close()
print("Saved whitney_bounds_paper3.png")
