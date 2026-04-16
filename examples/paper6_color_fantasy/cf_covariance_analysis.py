"""
Step 1: Multi-scale covariance and lag analysis for Color Fantasy.

For each meteorological input, at time scales {1h, 4h, 24h, 48h}:
  - Compute cross-correlation with each energy channel at lags 0…72h
  - Identify peak lag and peak correlation magnitude
  - Flag inputs with |r_max| < threshold at ALL scales as blind inputs

This tells us:
  (a) Which inputs actually influence which outputs
  (b) The characteristic lag (= approximate thermal time constant) for each coupling
  (c) Which inputs can be dropped from the encoder
"""
import numpy as np, pandas as pd, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

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

# Channels and inputs
CHANNELS = ['Acc. heating W','Electric cooling W','Domestic hot water W',
            'Galley steam demand W','HVAC aux W','Lighting facility W',
            'Equipment tenant W']
CHAN_SHORT = ['Heating','Cooling','DHW','Galley','HVAC','Lighting','Equipment']

INPUTS = ['Air temp [°C]','Relative humidity [%]','Total wind [m/s]',
          'Direct normal radiation [W/m2]',
          'Diffuse radition on horizontal surface [W/m2]','Cloud cover [%]']
INPUT_SHORT = ['T_amb','RH','Wind','DNI','Diffuse','Cloud']

# Convert channels to MW, remove seasonal trend for clean correlation
Y_raw = df[CHANNELS].values / 1e6
X_raw = df[INPUTS].values

# Remove slow seasonal trend (48h rolling mean) before computing lag correlations
# — otherwise the annual cycle dominates everything
def detrend_seasonal(x, window=48*7):
    """Remove 7-day rolling mean to isolate sub-weekly dynamics."""
    trend = pd.Series(x).rolling(window, center=True, min_periods=1).mean().values
    return x - trend

Y_dt = np.stack([detrend_seasonal(Y_raw[:,i]) for i in range(7)], axis=1)
X_dt = np.stack([detrend_seasonal(X_raw[:,i]) for i in range(6)], axis=1)

N = len(df)

# ── Time-scale averages ────────────────────────────────────────────────────────
def rolling_mean(arr, w):
    return pd.DataFrame(arr).rolling(w, min_periods=1).mean().values

scales = {'1h (raw)': 1, '4h avg': 4, '24h avg': 24, '48h avg': 48}

# ── Cross-correlation at each scale ───────────────────────────────────────────
MAX_LAG = 72   # hours
lags = np.arange(0, MAX_LAG+1)

print("Computing cross-correlations...")
# Results: [scale, input, channel, lag] → r value
results = {}  # (scale_name, input_idx, chan_idx) → (peak_r, peak_lag, r_at_lags)

for scale_name, w in scales.items():
    Xs = rolling_mean(X_dt, w)
    Ys = rolling_mean(Y_dt, w)
    # Normalise each column
    Xs_n = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
    Ys_n = (Ys - Ys.mean(0)) / (Ys.std(0) + 1e-9)
    
    for ii in range(6):
        for ci in range(7):
            r_vals = []
            for lag in lags:
                if lag == 0:
                    r = np.corrcoef(Xs_n[:,ii], Ys_n[:,ci])[0,1]
                else:
                    r = np.corrcoef(Xs_n[:-lag,ii], Ys_n[lag:,ci])[0,1]
                r_vals.append(r)
            r_vals = np.array(r_vals)
            peak_idx = np.argmax(np.abs(r_vals))
            results[(scale_name, ii, ci)] = (r_vals[peak_idx], lags[peak_idx], r_vals)

print("Done.")

# ── Figure 1: Peak |r| heatmap across scales ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Multi-scale Cross-Correlation: Meteorological Inputs vs Energy Channels\n'
             '(Seasonal trend removed; peak |r| at optimal lag 0–72h)',
             fontsize=12, fontweight='bold')

for ax, (scale_name, w) in zip(axes.flat, scales.items()):
    mat = np.zeros((6, 7))
    lag_mat = np.zeros((6, 7), dtype=int)
    for ii in range(6):
        for ci in range(7):
            peak_r, peak_lag, _ = results[(scale_name, ii, ci)]
            mat[ii, ci] = abs(peak_r)
            lag_mat[ii, ci] = peak_lag
    
    im = ax.imshow(mat, vmin=0, vmax=0.8, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(7)); ax.set_xticklabels(CHAN_SHORT, fontsize=9)
    ax.set_yticks(range(6)); ax.set_yticklabels(INPUT_SHORT, fontsize=9)
    ax.set_title(f'{scale_name}', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, label='peak |r|')
    
    # Annotate with lag in hours
    for ii in range(6):
        for ci in range(7):
            r_val = mat[ii,ci]
            lag_h = lag_mat[ii,ci]
            color = 'white' if r_val > 0.45 else 'black'
            ax.text(ci, ii, f'{r_val:.2f}\n{lag_h}h',
                    ha='center', va='center', fontsize=7, color=color)
    
    # Mark "blind" inputs (peak |r| < 0.10 for ALL channels)
    for ii in range(6):
        max_r = mat[ii].max()
        if max_r < 0.10:
            ax.annotate('← blind', xy=(6.5, ii), xytext=(6.6, ii),
                        fontsize=7, color='red', va='center')

plt.tight_layout()
fig.savefig(OUTDIR+'cf_covariance_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved cf_covariance_heatmap.png")

# ── Figure 2: Cross-correlation curves for the strongest couplings ────────────
# Show T_amb vs Heating, Cooling, HVAC at different scales (to reveal lags)
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Cross-Correlation Curves: T_amb → Thermal Channels\n'
             'Lag structure reveals thermal time constants',
             fontsize=11, fontweight='bold')

thermal_pairs = [
    (0, 0, 'T_amb → Heating'),
    (0, 1, 'T_amb → Cooling'),
    (0, 4, 'T_amb → HVAC'),
]
scale_colors = {'1h (raw)':'#1f77b4','4h avg':'#ff7f0e',
                '24h avg':'#2ca02c','48h avg':'#d62728'}

for ai, (ii, ci, title) in enumerate(thermal_pairs):
    ax = axes[0, ai]
    for scale_name, w in scales.items():
        _, _, r_vals = results[(scale_name, ii, ci)]
        ax.plot(lags, r_vals, color=scale_colors[scale_name],
                lw=2, label=scale_name, alpha=0.9)
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.axhline(0.1, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(-0.1, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Lag (hours)'); ax.set_ylabel('r')
    ax.set_title(title, fontweight='bold'); ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)
    
    # Mark peak lag
    _, _, r_vals_1h = results[('1h (raw)', ii, ci)]
    peak_lag = lags[np.argmax(np.abs(r_vals_1h))]
    ax.axvline(peak_lag, color='purple', lw=1.5, ls='--',
               label=f'peak lag={peak_lag}h')

# Bottom row: schedule-driven channels — should show no lag with T_amb
sched_pairs = [
    (0, 3, 'T_amb → Galley'),
    (0, 5, 'T_amb → Lighting'),
    (0, 6, 'T_amb → Equipment'),
]
for ai, (ii, ci, title) in enumerate(sched_pairs):
    ax = axes[1, ai]
    for scale_name, w in scales.items():
        _, _, r_vals = results[(scale_name, ii, ci)]
        ax.plot(lags, r_vals, color=scale_colors[scale_name],
                lw=2, label=scale_name, alpha=0.9)
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.fill_between(lags, -0.1, 0.1, alpha=0.1, color='green',
                    label='|r|<0.10 band')
    ax.set_xlabel('Lag (hours)'); ax.set_ylabel('r')
    ax.set_title(title, fontweight='bold'); ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
fig.savefig(OUTDIR+'cf_covariance_lags.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved cf_covariance_lags.png")

# ── Figure 3: DNI, RH, Wind, Cloud vs all channels ────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Other Meteorological Inputs: Are They Informative?\n'
             'Peak |r| at 24h scale vs each energy channel',
             fontsize=11, fontweight='bold')

other_inputs = [1, 2, 3, 4, 5]  # RH, Wind, DNI, Diffuse, Cloud
for ai, ii in enumerate(other_inputs[:5]):
    ax = axes.flat[ai]
    for scale_name, w in list(scales.items())[1:]:  # skip 1h for cleaner view
        mat_row = np.array([abs(results[(scale_name,ii,ci)][0]) for ci in range(7)])
        ax.bar(np.arange(7)+(list(scales.keys()).index(scale_name)-1)*0.25,
               mat_row, 0.25, label=scale_name,
               color=scale_colors[scale_name], alpha=0.8)
    ax.axhline(0.1, color='red', lw=1, ls='--', label='threshold 0.10')
    ax.set_xticks(range(7)); ax.set_xticklabels(CHAN_SHORT, rotation=30, fontsize=8)
    ax.set_ylabel('peak |r|'); ax.set_title(f'{INPUT_SHORT[ii]}', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 0.7)

# Last panel: summary table
ax = axes.flat[5]
ax.axis('off')
# Build summary: for each input, max |r| across all channels and scales
summary_rows = []
for ii, inp in enumerate(INPUT_SHORT):
    max_r = max(abs(results[(s,ii,ci)][0]) 
                for s in scales for ci in range(7))
    best_chan = max(range(7), 
                   key=lambda ci: max(abs(results[(s,ii,ci)][0]) for s in scales))
    best_scale = max(scales.keys(),
                     key=lambda s: abs(results[(s,ii,best_chan)][0]))
    best_lag = results[(best_scale, ii, best_chan)][1]
    keep = '✓ KEEP' if max_r >= 0.10 else '✗ DROP'
    summary_rows.append([inp, f'{max_r:.3f}', CHAN_SHORT[best_chan], 
                         best_scale.split()[0], f'{best_lag}h', keep])

tbl = ax.table(cellText=summary_rows,
               colLabels=['Input','Max |r|','Best chan','Scale','Lag','Decision'],
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.6)
for (r,c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2166ac'); cell.set_text_props(color='w',weight='bold')
    elif r > 0 and c == 5:
        if '✓' in cell.get_text().get_text():
            cell.set_facecolor('#d4f0d4')
        else:
            cell.set_facecolor('#fde8d8')
ax.set_title('Input reduction summary', fontsize=9, fontweight='bold', pad=15)

plt.tight_layout()
fig.savefig(OUTDIR+'cf_input_reduction.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved cf_input_reduction.png")

# ── Print numerical summary ───────────────────────────────────────────────────
print("\n" + "="*65)
print("INPUT REDUCTION SUMMARY")
print("="*65)
print(f"{'Input':10s} {'MaxR':6s} {'BestChan':10s} {'BestScale':10s} {'Lag':5s} {'Decision'}")
print("-"*65)
for row in summary_rows:
    print(f"{row[0]:10s} {row[1]:6s} {row[2]:10s} {row[3]:10s} {row[4]:5s} {row[5]}")

print("\nCHANNEL THERMAL LAG SUMMARY (T_amb, at 24h scale)")
print("-"*45)
for ci, chan in enumerate(CHAN_SHORT):
    peak_r, peak_lag, _ = results[('24h avg', 0, ci)]
    print(f"  T_amb → {chan:10s}: r={peak_r:+.3f}  lag={peak_lag:3d}h")

print("\nKEY FINDINGS:")
# Auto-summarise
blind = [INPUT_SHORT[ii] for ii in range(6) 
         if max(abs(results[(s,ii,ci)][0]) 
                for s in scales for ci in range(7)) < 0.10]
thermal = [(CHAN_SHORT[ci], results[('24h avg',0,ci)][1])
           for ci in range(7) if abs(results[('24h avg',0,ci)][0]) > 0.3]
schedule = [CHAN_SHORT[ci] for ci in range(7)
            if max(abs(results[(s,0,ci)][0]) for s in scales) < 0.15]
print(f"  Blind inputs (|r|<0.10 everywhere): {blind if blind else 'None'}")
print(f"  Thermal channels (|r|>0.3 with T_amb): {thermal}")
print(f"  Schedule channels (|r_T_amb|<0.15): {schedule}")
