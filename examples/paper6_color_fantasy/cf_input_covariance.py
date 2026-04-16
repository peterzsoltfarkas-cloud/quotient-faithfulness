"""
Input-to-input partial cross-covariance for Color Fantasy.
Same method: control for time-of-day and season, then compute
cross-correlations between all pairs of meteorological inputs.

This reveals:
  - Which inputs are redundant (highly correlated → carry same information)
  - Independent dimensions in the input space
  - Whether PCA on inputs is warranted before building the encoder
"""
import numpy as np, pandas as pd, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

OUTDIR = '/mnt/user-data/outputs/figures/'

# ── Load data ─────────────────────────────────────────────────────────────────
ida = pd.read_excel(
    '/mnt/user-data/uploads/Color_Fantasy_IDA_ICE_results_-_August_Brækken.xlsx')
wx  = pd.read_excel(
    '/mnt/user-data/uploads/Color_Fantasy_weather_data_-_August_Brækken.xlsx')

df = ida[ida.iloc[:,0]==1].copy().reset_index(drop=True)
df['hour']    = df['Time hours'] % 24
df['day']     = (df['Time hours'] // 24).astype(int)
df['hour_int']= df['Time hours'].apply(lambda h: int(np.ceil(h)))
wx['hour_int']= wx['Time [h]'].astype(int)
df = df.merge(wx, on='hour_int', how='left').ffill()

INPUTS     = ['Air temp [°C]','Relative humidity [%]','Total wind [m/s]',
              'Direct normal radiation [W/m2]',
              'Diffuse radition on horizontal surface [W/m2]','Cloud cover [%]']
INPUT_SHORT= ['T_amb','RH','Wind','DNI','Diffuse','Cloud']

CHANNELS   = ['Acc. heating W','Electric cooling W','Domestic hot water W',
              'Galley steam demand W','HVAC aux W','Lighting facility W',
              'Equipment tenant W']
CHAN_SHORT  = ['Heating','Cooling','DHW','Galley','HVAC','Lighting','Equipment']

X_raw = df[INPUTS].values
Y_raw = df[CHANNELS].values / 1e6

hour = df['hour'].values
day  = df['day'].values

# Confounder matrix
Z = np.column_stack([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                     np.sin(2*np.pi*day/365),  np.cos(2*np.pi*day/365)])

def partial_res(v, Z):
    return v - LinearRegression().fit(Z, v).predict(Z)

print("Computing partial residuals...")
X_res = np.column_stack([partial_res(X_raw[:,i], Z) for i in range(6)])
Y_res = np.column_stack([partial_res(Y_raw[:,i], Z) for i in range(7)])

N = len(df)
SIG = 2/np.sqrt(N)
print(f"  N={N},  significance threshold |r|>{SIG:.4f}")

# ── 1. Contemporaneous input-input partial correlation matrix ─────────────────
SCALES = [('1h',1),('4h',4),('24h',24),('48h',48)]
LAGS   = [0, 2, 4, 6, 12, 24, 36, 48, 72]

def rolling_mean(arr, w):
    if w == 1: return arr
    return pd.DataFrame(arr).rolling(w, min_periods=1).mean().values

# Full input×input cross-correlation at all scales and lags
ii_results = {}   # (scale, i, j) → {lag: r}
for sname, w in SCALES:
    Xs = rolling_mean(X_res, w)
    Xn = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
    for i in range(6):
        for j in range(6):
            lag_r = {}
            for lag in LAGS:
                if lag == 0:
                    r = np.corrcoef(Xn[:,i], Xn[:,j])[0,1]
                else:
                    r = np.corrcoef(Xn[:-lag,i], Xn[lag:,j])[0,1]
                lag_r[lag] = float(r)
            peak_r   = max(lag_r.values(), key=abs)
            peak_lag = max(lag_r.keys(), key=lambda l: abs(lag_r[l]))
            ii_results[(sname,i,j)] = {'lag_r':lag_r,'peak_r':peak_r,'peak_lag':peak_lag}

# ── Print full numerical table ────────────────────────────────────────────────
print("\n" + "="*70)
print("INPUT × INPUT PARTIAL CROSS-CORRELATION")
print("Controlling for time-of-day and season")
print(f"Significance: |r| > {SIG:.3f}  [* marks significant contemporaneous r]")
print("="*70)

for sname, w in SCALES:
    print(f"\n{'─'*70}")
    print(f"SCALE: {sname}")
    print(f"{'─'*70}")
    header = f"{'':8s}" + "".join(f"{s:>12s}" for s in INPUT_SHORT)
    print(header)
    print("─"*len(header))
    for i, inp_i in enumerate(INPUT_SHORT):
        row = f"{inp_i:8s}"
        for j, inp_j in enumerate(INPUT_SHORT):
            r   = ii_results[(sname,i,j)]['peak_r']
            lag = ii_results[(sname,i,j)]['peak_lag']
            sig = '*' if abs(r)>SIG else ' '
            if i == j:
                row += f"   1.000     "
            else:
                row += f"  {r:+.3f}@{lag:2d}h{sig}"
        print(row)

# ── Contemporaneous correlation matrix (lag=0) for PCA ───────────────────────
print("\n" + "="*70)
print("CONTEMPORANEOUS PARTIAL r MATRIX (lag=0, 24h scale)")
print("="*70)
Xs_24 = rolling_mean(X_res, 24)
Xn_24 = (Xs_24 - Xs_24.mean(0)) / (Xs_24.std(0)+1e-9)
R_24  = np.corrcoef(Xn_24.T)
header = f"{'':8s}" + "".join(f"{s:>8s}" for s in INPUT_SHORT)
print(header)
for i, inp in enumerate(INPUT_SHORT):
    row = f"{inp:8s}" + "".join(f"  {R_24[i,j]:+.3f} " for j in range(6))
    print(row)

# Eigenvalues
eigvals, eigvecs = np.linalg.eigh(R_24)
eigvals = eigvals[::-1]; eigvecs = eigvecs[:,::-1]
print(f"\nEigenvalues of R (variance explained by each PC):")
cumvar = 0
for k,(ev,frac) in enumerate(zip(eigvals, eigvals/eigvals.sum())):
    cumvar += frac
    print(f"  PC{k+1}: λ={ev:.3f}  ({frac*100:.1f}%  cumulative {cumvar*100:.1f}%)")

print("\nEigenvectors (PC loadings):")
print(f"{'':8s}" + "".join(f"  PC{k+1} " for k in range(6)))
for i, inp in enumerate(INPUT_SHORT):
    row = f"{inp:8s}" + "".join(f"  {eigvecs[i,k]:+.3f}" for k in range(6))
    print(row)

# ── Also: input-input vs input-output comparison ─────────────────────────────
print("\n" + "="*70)
print("INDEPENDENCE SUMMARY: which inputs are highly collinear?")
print("|r|>0.40 contemporaneous pairs at 24h scale")
print("="*70)
for i in range(6):
    for j in range(i+1,6):
        r = R_24[i,j]
        if abs(r) > 0.30:
            flag = "HIGH" if abs(r)>0.5 else "MOD"
            print(f"  {flag}: {INPUT_SHORT[i]:8s} ↔ {INPUT_SHORT[j]:8s}  r={r:+.3f}")


# ── FIGURES ────────────────────────────────────────────────────────────────────

# Figure 1: Input×input contemporaneous partial r, all four scales
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle('Input × Input Partial Cross-Correlation\n'
             'Controlling for time-of-day and season\n'
             'Values = peak |r| at optimal lag 0–72h (signed)',
             fontsize=12, fontweight='bold')

for ax, (sname, w) in zip(axes.flat, SCALES):
    mat = np.zeros((6,6))
    mat_lag = np.zeros((6,6), dtype=int)
    for i in range(6):
        for j in range(6):
            if i==j:
                mat[i,j] = 1.0
            else:
                mat[i,j]     = ii_results[(sname,i,j)]['peak_r']
                mat_lag[i,j] = ii_results[(sname,i,j)]['peak_lag']

    im = ax.imshow(mat, vmin=-0.85, vmax=0.85, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(6)); ax.set_xticklabels(INPUT_SHORT, fontsize=9)
    ax.set_yticks(range(6)); ax.set_yticklabels(INPUT_SHORT, fontsize=9)
    ax.set_title(f'Scale: {sname}', fontweight='bold', fontsize=10)
    plt.colorbar(im, ax=ax, label='partial r (signed)')

    for i in range(6):
        for j in range(6):
            color = 'white' if abs(mat[i,j]) > 0.5 else 'black'
            if i == j:
                ax.text(j, i, '1.00', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')
            else:
                lag_h = mat_lag[i,j]
                ax.text(j, i, f'{mat[i,j]:+.2f}\n{lag_h}h',
                        ha='center', va='center', fontsize=7, color=color,
                        fontweight='bold' if abs(mat[i,j])>0.30 else 'normal')

plt.tight_layout()
fig.savefig(OUTDIR+'cf_input_input_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved cf_input_input_heatmap.png")

# Figure 2: PCA scree + loadings
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('PCA of Partial-Residual Inputs (24h scale)\n'
             'How many independent dimensions are in the meteorological forcing?',
             fontsize=11, fontweight='bold')

# Scree plot
ax = axes[0]
cumvar = np.cumsum(eigvals/eigvals.sum())*100
ax.bar(range(1,7), eigvals/eigvals.sum()*100, color='#2166ac', alpha=0.8)
ax.plot(range(1,7), cumvar, 'ro-', lw=2, ms=7, label='Cumulative %')
ax.axhline(80, color='gray', ls='--', lw=1.5, label='80% threshold')
ax.axhline(95, color='green', ls='--', lw=1.5, label='95% threshold')
for k in range(6):
    ax.text(k+1, eigvals[k]/eigvals.sum()*100+0.5, f'{eigvals[k]/eigvals.sum()*100:.1f}%',
            ha='center', fontsize=8.5)
ax.set_xlabel('Principal Component'); ax.set_ylabel('Variance explained (%)')
ax.set_title('Scree plot: variance per PC', fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

# Loadings heatmap
ax = axes[1]
loadings = eigvecs[:,:4]  # top 4 PCs
im = ax.imshow(loadings, vmin=-0.8, vmax=0.8, cmap='RdBu_r', aspect='auto')
ax.set_xticks(range(4)); ax.set_xticklabels([f'PC{k+1}' for k in range(4)])
ax.set_yticks(range(6)); ax.set_yticklabels(INPUT_SHORT)
ax.set_title('PC loadings (top 4)', fontweight='bold')
plt.colorbar(im, ax=ax)
for i in range(6):
    for j in range(4):
        color = 'white' if abs(loadings[i,j])>0.45 else 'black'
        ax.text(j, i, f'{loadings[i,j]:+.2f}',
                ha='center', va='center', fontsize=9, color=color)

# Key lag-correlation pairs visualisation
ax = axes[2]
# Show the strongest off-diagonal input pairs at 24h scale
pairs_to_show = []
for i in range(6):
    for j in range(i+1,6):
        r = ii_results[('24h',i,j)]['peak_r']
        lag = ii_results[('24h',i,j)]['peak_lag']
        if abs(r) > 0.20:
            pairs_to_show.append((abs(r), r, lag, INPUT_SHORT[i], INPUT_SHORT[j]))
pairs_to_show.sort(reverse=True)

colors = ['#d62728' if r>0 else '#1f77b4' for _,r,_,_,_ in pairs_to_show]
labels = [f'{a}↔{b}\nlag={lag}h' for _,_,lag,a,b in pairs_to_show]
vals   = [abs(r) for _,r,_,_,_ in pairs_to_show]
bars   = ax.barh(range(len(vals)), vals, color=colors, alpha=0.85)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('peak |r| (24h scale)')
ax.set_title('Strongest input–input\ncouplings (|r|>0.20)', fontweight='bold')
ax.axvline(0.40, color='gray', ls='--', lw=1, alpha=0.7, label='|r|=0.40')
ax.axvline(0.60, color='red',  ls='--', lw=1, alpha=0.7, label='|r|=0.60')
ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)
for bar, v in zip(bars, vals):
    ax.text(v+0.01, bar.get_y()+bar.get_height()/2, f'{v:.2f}',
            va='center', fontsize=8)

plt.tight_layout()
fig.savefig(OUTDIR+'cf_input_pca.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved cf_input_pca.png")

# Figure 3: Lag profiles for the strongly coupled input pairs
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Input × Input Lag Profiles (24h scale)\n'
             'Partial correlations after seasonal + diurnal removal',
             fontsize=11, fontweight='bold')

strong_pairs = [(i,j) for i in range(6) for j in range(i+1,6)
                if abs(ii_results[('24h',i,j)]['peak_r']) > 0.20]
strong_pairs.sort(key=lambda ij: abs(ii_results[('24h',ij[0],ij[1])]['peak_r']),
                   reverse=True)

scale_cols = {'1h':'#1f77b4','4h':'#ff7f0e','24h':'#2ca02c','48h':'#d62728'}

for ax, (i,j) in zip(axes.flat, strong_pairs[:6]):
    for sname, w in SCALES:
        lr = ii_results[(sname,i,j)]['lag_r']
        lags_p = sorted(lr.keys())
        ax.plot(lags_p, [lr[l] for l in lags_p],
                color=scale_cols[sname], lw=2, marker='o', ms=4,
                label=sname, alpha=0.9)
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.fill_between(lags_p, -SIG, SIG, alpha=0.1, color='green')
    peak_r   = ii_results[('24h',i,j)]['peak_r']
    peak_lag = ii_results[('24h',i,j)]['peak_lag']
    ax.set_title(f'{INPUT_SHORT[i]} ↔ {INPUT_SHORT[j]}\npeak r={peak_r:+.3f} @{peak_lag}h',
                 fontweight='bold', fontsize=9)
    ax.set_xlabel('Lag (h)'); ax.set_ylabel('partial r')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    ax.set_ylim(-0.85, 0.85)

plt.tight_layout()
fig.savefig(OUTDIR+'cf_input_input_lags.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved cf_input_input_lags.png")
