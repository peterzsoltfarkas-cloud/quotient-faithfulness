"""
Proper partial covariance analysis for Color Fantasy.
Controls for BOTH time-of-day and season before computing cross-correlations.

Method for each (input, channel, lag) triple:
  1. Regress sin(2πh/24), cos(2πh/24), sin(2πd/365), cos(2πd/365)
     out of BOTH the input x(t) and the output y(t+lag)
  2. Compute Pearson r on the residuals
  3. Confidence interval: |r| > 2/√N is 95% significant

Time scales tested: raw 1h, 4h rolling average, 24h rolling average, 48h rolling average
Lags tested: 0h, 2h, 4h, 6h, 12h, 24h, 36h, 48h, 72h
"""
import numpy as np, pandas as pd, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

OUTDIR = '/mnt/user-data/outputs/figures/'

# ── Load data ─────────────────────────────────────────────────────────────────
ida = pd.read_excel(
    '/mnt/user-data/uploads/Color_Fantasy_IDA_ICE_results_-_August_Brækken.xlsx')
wx  = pd.read_excel(
    '/mnt/user-data/uploads/Color_Fantasy_weather_data_-_August_Brækken.xlsx')

df = ida[ida.iloc[:,0]==1].copy().reset_index(drop=True)
df['hour']    = df['Time hours'] % 24
df['day']     = (df['Time hours'] // 24).astype(int)
df['in_port'] = ((df['hour']>=10)&(df['hour']<14)
                 &(df['Propulsion W']<1e6)).astype(int)
df['hour_int']= df['Time hours'].apply(lambda h: int(np.ceil(h)))
wx['hour_int']= wx['Time [h]'].astype(int)
df = df.merge(wx, on='hour_int', how='left').ffill()

CHANNELS  = ['Acc. heating W','Electric cooling W','Domestic hot water W',
             'Galley steam demand W','HVAC aux W','Lighting facility W',
             'Equipment tenant W']
CHAN_SHORT = ['Heating','Cooling','DHW','Galley','HVAC','Lighting','Equipment']

INPUTS     = ['Air temp [°C]','Relative humidity [%]','Total wind [m/s]',
              'Direct normal radiation [W/m2]',
              'Diffuse radition on horizontal surface [W/m2]','Cloud cover [%]']
INPUT_SHORT= ['T_amb','RH','Wind','DNI','Diffuse','Cloud']

Y_raw = df[CHANNELS].values / 1e6   # MW
X_raw = df[INPUTS].values

# Confounder matrix: sin/cos hour + sin/cos day
hour = df['hour'].values
day  = df['day'].values
Z_confounders = np.column_stack([
    np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
    np.sin(2*np.pi*day/365), np.cos(2*np.pi*day/365)
])

# ── Partial residuals: remove time-of-day and season ──────────────────────────
def partial_residual(v, Z):
    """Regress Z out of v, return residuals."""
    lr = LinearRegression().fit(Z, v)
    return v - lr.predict(Z)

print("Computing partial residuals...")
X_res = np.column_stack([partial_residual(X_raw[:,i], Z_confounders)
                          for i in range(6)])
Y_res = np.column_stack([partial_residual(Y_raw[:,i], Z_confounders)
                          for i in range(7)])
print(f"  N={len(df)} hourly observations")
print(f"  95% significance threshold: |r| > {2/np.sqrt(len(df)):.4f}")

SIG_THRESHOLD = 2/np.sqrt(len(df))

# ── Multi-scale partial cross-correlation ────────────────────────────────────
def rolling_mean(arr, w):
    if w == 1: return arr
    return pd.DataFrame(arr).rolling(w, min_periods=1).mean().values

SCALES = [('1h',1), ('4h',4), ('24h',24), ('48h',48)]
LAGS   = [0, 2, 4, 6, 12, 24, 36, 48, 72]

# Full results table: [scale, input, channel] → dict of lag→r
results = {}
for (sname, w) in SCALES:
    Xs = rolling_mean(X_res, w)
    Ys = rolling_mean(Y_res, w)
    # Re-normalise after averaging
    Xs_n = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
    Ys_n = (Ys - Ys.mean(0)) / (Ys.std(0) + 1e-9)
    for ii in range(6):
        for ci in range(7):
            lag_r = {}
            for lag in LAGS:
                if lag == 0:
                    r = np.corrcoef(Xs_n[:,ii], Ys_n[:,ci])[0,1]
                else:
                    r = np.corrcoef(Xs_n[:-lag,ii], Ys_n[lag:,ci])[0,1]
                lag_r[lag] = float(r)
            peak_r  = max(lag_r.values(), key=abs)
            peak_lag= max(lag_r.keys(), key=lambda l: abs(lag_r[l]))
            results[(sname, ii, ci)] = {
                'lag_r': lag_r, 'peak_r': peak_r, 'peak_lag': peak_lag
            }

# ── FULL NUMERICAL TABLE (what Peter asked for) ───────────────────────────────
print("\n" + "="*80)
print("FULL PARTIAL CROSS-CORRELATION TABLE")
print("Controlling for: sin/cos(hour), sin/cos(day/365)")
print(f"Significance: |r| > {SIG_THRESHOLD:.3f}  [marked with *]")
print("="*80)

for sname, w in SCALES:
    print(f"\n{'─'*80}")
    print(f"TIME SCALE: {sname} rolling average")
    print(f"{'─'*80}")
    # Header
    header = f"{'Input':10s}" + "".join(f" {c:>14s}" for c in CHAN_SHORT)
    print(header)
    print("─"*len(header))
    for ii, inp in enumerate(INPUT_SHORT):
        row = f"{inp:10s}"
        for ci in range(7):
            res = results[(sname, ii, ci)]
            r   = res['peak_r']
            lag = res['peak_lag']
            sig = '*' if abs(r) > SIG_THRESHOLD else ' '
            row += f"  {r:+.3f}@{lag:2d}h{sig}"
        print(row)

# ── Lag-profile table: for each channel, which input has peak r and at what lag?
print("\n" + "="*80)
print("CHANNEL SUMMARY: dominant input, peak partial r, optimal lag (24h scale)")
print("="*80)
print(f"{'Channel':12s} {'Dom.Input':10s} {'Peak r':8s} {'Peak lag':10s} {'2nd input':10s} {'2nd r':8s}")
print("─"*70)
for ci, chan in enumerate(CHAN_SHORT):
    # Find best input at 24h scale
    scale_res = [(INPUT_SHORT[ii],
                  results[('24h',ii,ci)]['peak_r'],
                  results[('24h',ii,ci)]['peak_lag'])
                 for ii in range(6)]
    scale_res.sort(key=lambda x: abs(x[1]), reverse=True)
    dom  = scale_res[0]; sec = scale_res[1]
    print(f"{chan:12s} {dom[0]:10s} {dom[1]:+8.3f} {dom[2]:4d}h       "
          f"{sec[0]:10s} {sec[1]:+8.3f}")

# ── Channel grouping by physics character (partial r reveals true structure)
print("\n" + "="*80)
print("PHYSICS CHARACTER REVEALED BY PARTIAL CORRELATIONS (24h scale)")
print("='Externally forced' if |r_T_amb|>0.20 after controlling for season")
print("="*80)
for ci, chan in enumerate(CHAN_SHORT):
    r_T = results[('24h',0,ci)]['peak_r']
    r_DNI = results[('24h',3,ci)]['peak_r']
    r_lag = results[('24h',0,ci)]['peak_lag']
    if abs(r_T) > 0.20:
        char = f"THERMALLY FORCED  (r_T={r_T:+.3f}, lag={r_lag}h)"
    elif abs(r_DNI) > 0.20:
        char = f"SOLAR FORCED  (r_DNI={r_DNI:+.3f})"
    else:
        char = f"INTERNALLY GENERATED  (|r_T|={abs(r_T):.3f}, below 0.20)"
    print(f"  {chan:12s}: {char}")


# ── Visual tables ─────────────────────────────────────────────────────────────
# Figure: 4-panel heatmap, one per scale, partial r values
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Partial Cross-Correlation: Meteorological Inputs → Energy Channels\n'
             'Controlling for time-of-day AND season  |  peak |r| at optimal lag 0–72h',
             fontsize=12, fontweight='bold')

for ax, (sname, w) in zip(axes.flat, SCALES):
    mat_r   = np.zeros((6, 7))
    mat_lag = np.zeros((6, 7), dtype=int)
    for ii in range(6):
        for ci in range(7):
            r   = results[(sname, ii, ci)]['peak_r']
            lag = results[(sname, ii, ci)]['peak_lag']
            mat_r[ii, ci]   = r          # keep sign
            mat_lag[ii, ci] = lag

    im = ax.imshow(mat_r, vmin=-0.75, vmax=0.75, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(7)); ax.set_xticklabels(CHAN_SHORT, fontsize=9)
    ax.set_yticks(range(6)); ax.set_yticklabels(INPUT_SHORT, fontsize=9)
    ax.set_title(f'Scale: {sname}  (seasonal + diurnal removed)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='partial r (signed)')

    for ii in range(6):
        for ci in range(7):
            r_val = mat_r[ii, ci]
            lag_h = mat_lag[ii, ci]
            sig   = abs(r_val) > SIG_THRESHOLD
            color = 'white' if abs(r_val) > 0.35 else 'black'
            # Bold if significant AND |r|>0.2, italic otherwise
            w_str = 'bold' if abs(r_val) > 0.20 else 'normal'
            ax.text(ci, ii, f'{r_val:+.2f}\n{lag_h}h',
                    ha='center', va='center', fontsize=6.5,
                    color=color, fontweight=w_str)
            # Mark non-significant (|r|<threshold) with grey box
            if not sig:
                ax.add_patch(plt.Rectangle((ci-0.5, ii-0.5), 1, 1,
                             fill=True, facecolor='white', alpha=0.5))

plt.tight_layout()
fig.savefig(OUTDIR+'cf_partial_covariance_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved cf_partial_covariance_heatmap.png")

# ── Figure 2: Lag profiles for the clearest couplings (24h scale)
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Partial Cross-Correlation Lag Profiles (24h scale)\n'
             'Seasonal and diurnal confounders removed — residual coupling only',
             fontsize=11, fontweight='bold')

# Choose the most interesting pairs
pairs = [
    (0, 0, 'T_amb → Heating\n(thermal, lag~2h)'),
    (0, 1, 'T_amb → Cooling\n(thermal, lag~6h)'),
    (0, 2, 'T_amb → DHW\n(mixed, lag~0h)'),
    (3, 1, 'DNI → Cooling\n(solar gain, lag~6h)'),
    (4, 2, 'Diffuse → DHW\n(lag~12h)'),
    (4, 5, 'Diffuse → Lighting\n(daylighting, lag~12h)'),
    (0, 3, 'T_amb → Galley\n(lag~36h: suspicious?)'),
    (1, 0, 'RH → Heating\n(infiltration? lag~0h)'),
    (5, 1, 'Cloud → Cooling\n(weakest coupling)'),
]
scale_colors = {'1h':'#1f77b4','4h':'#ff7f0e','24h':'#2ca02c','48h':'#d62728'}

for ax, (ii, ci, title) in zip(axes.flat, pairs):
    for sname, _ in SCALES:
        lag_r = results[(sname, ii, ci)]['lag_r']
        lags_plot = sorted(lag_r.keys())
        r_plot    = [lag_r[l] for l in lags_plot]
        ax.plot(lags_plot, r_plot, color=scale_colors[sname],
                lw=2, label=sname, alpha=0.85, marker='o', ms=4)
    
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.axhline( SIG_THRESHOLD, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(-SIG_THRESHOLD, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.fill_between(lags_plot, -SIG_THRESHOLD, SIG_THRESHOLD,
                    alpha=0.1, color='green', label='not sig.')
    ax.set_xlabel('Lag (h)'); ax.set_ylabel('partial r')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(fontsize=6.5, ncol=2); ax.grid(True, alpha=0.2)
    ax.set_ylim(-0.85, 0.85)

plt.tight_layout()
fig.savefig(OUTDIR+'cf_partial_lag_profiles.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved cf_partial_lag_profiles.png")

# ── Summary table figure ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
ax.axis('off')

# Build the comprehensive summary table for Peter
rows = []
for ci, chan in enumerate(CHAN_SHORT):
    row = [chan]
    for sname, _ in SCALES:
        # Show all 6 inputs at this scale
        r_vals = [(INPUT_SHORT[ii], results[(sname,ii,ci)]['peak_r'],
                   results[(sname,ii,ci)]['peak_lag']) for ii in range(6)]
        r_vals.sort(key=lambda x: abs(x[1]), reverse=True)
        top = r_vals[0]
        row.append(f"{top[0]}:{top[1]:+.2f}@{top[2]}h")
    rows.append(row)

col_labels = ['Channel', 'Top driver\n(1h scale)', 'Top driver\n(4h scale)',
              'Top driver\n(24h scale)', 'Top driver\n(48h scale)']
tbl = ax.table(cellText=rows, colLabels=col_labels,
               cellLoc='center', loc='center', colWidths=[0.12,0.22,0.22,0.22,0.22])
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.0)
for (r,c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2166ac'); cell.set_text_props(color='w', weight='bold')
    elif r > 0:
        try:
            val = cell.get_text().get_text()
            r_num = abs(float(val.split(':')[1].split('@')[0]))
            if 'T_amb' in val and r_num > 0.3:
                cell.set_facecolor('#fde8d8')
            elif 'DNI' in val and r_num > 0.3:
                cell.set_facecolor('#fff3cd')
            elif r_num < 0.15:
                cell.set_facecolor('#f7f7f7')
        except (IndexError, ValueError):
            pass
ax.set_title('Top meteorological driver per channel per time scale\n'
             '(partial r after removing time-of-day and season)',
             fontsize=10, fontweight='bold', pad=20)

plt.tight_layout()
fig.savefig(OUTDIR+'cf_partial_summary_table.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved cf_partial_summary_table.png")
