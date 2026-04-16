"""
Thermal delay estimation per orthogonal state.

States (strictly non-overlapping, dead zone enforced):
  State 2: Full heating    T_wc <= 5°C
  State 1: Cabin heating   5°C < T_wc <= 21°C
  State 0: Dead zone       21°C < T_amb < 24°C   [no heating, no cooling]
  State 3: Cooling         T_amb >= 24°C

For each state, fit a FIR model per relevant channel:
  Q_channel(t) ≈ Σₖ βₖ · forcing(t-k)
to recover the impulse response (true delay distribution).

Forcing variables per state:
  States 1+2: T_wc (windchill)
  State 3:    T_amb (no windchill needed above 24°C)
  All states: DNI, Diffuse (radiation, independent)
"""
import numpy as np, pandas as pd, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression

OUTDIR = '/mnt/user-data/outputs/figures/'

# ── Load ───────────────────────────────────────────────────────────────────────
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

T    = df['Air temp [°C]'].values.astype(float)
V    = df['Total wind [m/s]'].values.astype(float)
DNI  = df['Direct normal radiation [W/m2]'].values.astype(float)
Dif  = df['Diffuse radition on horizontal surface [W/m2]'].values.astype(float)
RH   = df['Relative humidity [%]'].values.astype(float)
hour = df['hour'].values
day  = df['day'].values
port = df['in_port'].values.astype(float)

# ── Windchill ─────────────────────────────────────────────────────────────────
V_kmh = V * 3.6
T_wc  = np.where(T <= 10,
    13.12 + 0.6215*T - 11.37*(V_kmh**0.16) + 0.3965*T*(V_kmh**0.16), T)

# ── ORTHOGONAL states — strict boundaries, dead zone between 21 and 24 ────────
#   State 2: T_wc <= 5
#   State 1: 5 < T_wc <= 21
#   State 0: 21 < T_amb < 24   (dead zone: no heating, no cooling)
#   State 3: T_amb >= 24
STATE2 = T_wc  <= 5
STATE1 = (T_wc  >  5) & (T_wc <= 21)
STATE0 = (T     > 21) & (T    <  24)   # dead zone: T_amb (not windchill)
STATE3 = T     >= 24

# Verify strict orthogonality
total = STATE1.sum() + STATE2.sum() + STATE0.sum() + STATE3.sum()
overlap = len(T) - total
assert overlap == 0, f"States overlap by {overlap} hours — check logic"
print("State orthogonality: OK (zero overlap, full coverage)")

print(f"\nState distribution (N={len(T)}):")
for name, mask in [('State 2: Full heating  (T_wc≤5°C)',    STATE2),
                   ('State 1: Cabin heating (5<T_wc≤21°C)', STATE1),
                   ('State 0: Dead zone     (21<T<24°C)',    STATE0),
                   ('State 3: Cooling       (T≥24°C)',       STATE3)]:
    print(f"  {name}: {mask.sum():5d} h  ({mask.sum()/len(T)*100:.1f}%)")

# ── Remove confounders within each state ─────────────────────────────────────
# Confounder: time-of-day + port/sea (NOT season — we want to see seasonal lags)
def partial_resid(series, mask, extra_features=None):
    """Remove diurnal and port/sea cycle within a state mask."""
    h = hour[mask]; p = port[mask]
    Z = np.column_stack([np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24), p])
    if extra_features is not None:
        Z = np.column_stack([Z, extra_features])
    v = series[mask]
    return v - LinearRegression().fit(Z, v).predict(Z)

# ── FIR delay estimation ───────────────────────────────────────────────────────
MAX_LAG = 72
LAGS    = np.arange(MAX_LAG + 1)

def lag_matrix(x, max_lag):
    """Build (N-max_lag) × (max_lag+1) matrix of x[t], x[t-1], ..., x[t-max_lag]."""
    N = len(x) - max_lag
    return np.column_stack([x[max_lag - k : max_lag - k + N] for k in LAGS])

def fit_fir(forcing_r, output_r, alpha=10.0):
    """
    Ridge FIR: output[t] ~ Σₖ βₖ · forcing[t-k]
    Returns (coefficients, R²)
    Coefficients = estimated impulse response h[0..MAX_LAG]
    """
    X = lag_matrix(forcing_r, MAX_LAG)
    y = output_r[MAX_LAG:]
    model = Ridge(alpha=alpha, fit_intercept=False).fit(X, y)
    return model.coef_, model.score(X, y)

def delay_stats(h):
    """
    From impulse response h[k], compute:
      peak_lag: lag of largest |h|
      tau_eff:  energy-weighted mean lag (centre of mass of |h|)
      lag_50:   lag where cumulative |h| reaches 50% of total
      lag_90:   lag where cumulative |h| reaches 90% of total
    """
    h_abs = np.abs(h)
    if h_abs.sum() < 1e-10:
        return dict(peak=0, tau=0, lag50=0, lag90=0, sign=0)
    peak_lag = int(np.argmax(h_abs))
    tau_eff  = float(np.dot(LAGS, h_abs) / h_abs.sum())
    cum      = np.cumsum(h_abs) / h_abs.sum()
    lag50    = int(LAGS[np.searchsorted(cum, 0.50)])
    lag90    = int(LAGS[np.searchsorted(cum, 0.90)])
    sign     = float(np.sign(h[peak_lag]))
    return dict(peak=peak_lag, tau=tau_eff, lag50=lag50, lag90=lag90, sign=sign)

# ── State 1: Cabin heating ────────────────────────────────────────────────────
print("\n" + "="*65)
print("STATE 1 — Cabin heating (5°C < T_wc ≤ 21°C)")
print("="*65)

mask = STATE1
twc_r1  = partial_resid(T_wc, mask)
dni_r1  = partial_resid(DNI,  mask)
dif_r1  = partial_resid(Dif,  mask)
qh_r1   = partial_resid(df['Acc. heating W'].values/1e6, mask)
qhvac_r1= partial_resid(df['HVAC aux W'].values/1e6, mask)
qdew_r1 = partial_resid(df['Domestic hot water W'].values/1e6, mask)

fir_results = {}
for ch_name, y_r, forcing_name, x_r in [
    ('Q_H   ← T_wc',   qh_r1,   'T_wc',   twc_r1),
    ('Q_H   ← DNI',    qh_r1,   'DNI',    dni_r1),
    ('Q_H   ← Diffuse',qh_r1,   'Diffuse',dif_r1),
    ('HVAC  ← T_wc',   qhvac_r1,'T_wc',   twc_r1),
    ('DHW   ← T_wc',   qdew_r1, 'T_wc',   twc_r1),
    ('DHW   ← Diffuse',qdew_r1, 'Diffuse',dif_r1),
]:
    coef, r2 = fit_fir(x_r, y_r)
    ds = delay_stats(coef)
    fir_results[('S1', ch_name)] = (coef, r2, ds)
    print(f"  {ch_name:22s}: R²={r2:.3f}  sign={ds['sign']:+.0f}  "
          f"peak={ds['peak']:3d}h  τ_eff={ds['tau']:4.1f}h  "
          f"50%={ds['lag50']:3d}h  90%={ds['lag90']:3d}h")

# ── State 2: Full heating ─────────────────────────────────────────────────────
print("\n" + "="*65)
print("STATE 2 — Full heating (T_wc ≤ 5°C)")
print("="*65)

mask = STATE2
twc_r2   = partial_resid(T_wc, mask)
dni_r2   = partial_resid(DNI,  mask)
dif_r2   = partial_resid(Dif,  mask)
qh_r2    = partial_resid(df['Acc. heating W'].values/1e6, mask)
qhvac_r2 = partial_resid(df['HVAC aux W'].values/1e6, mask)

for ch_name, y_r, forcing_name, x_r in [
    ('Q_H   ← T_wc',   qh_r2,   'T_wc',   twc_r2),
    ('Q_H   ← DNI',    qh_r2,   'DNI',    dni_r2),
    ('HVAC  ← T_wc',   qhvac_r2,'T_wc',   twc_r2),
]:
    coef, r2 = fit_fir(x_r, y_r)
    ds = delay_stats(coef)
    fir_results[('S2', ch_name)] = (coef, r2, ds)
    print(f"  {ch_name:22s}: R²={r2:.3f}  sign={ds['sign']:+.0f}  "
          f"peak={ds['peak']:3d}h  τ_eff={ds['tau']:4.1f}h  "
          f"50%={ds['lag50']:3d}h  90%={ds['lag90']:3d}h")

# ── State 3: Cooling ───────────────────────────────────────────────────────────
print("\n" + "="*65)
print("STATE 3 — Cooling (T_amb ≥ 24°C)")
print("="*65)

mask = STATE3
if mask.sum() > MAX_LAG + 50:
    tamb_r3 = partial_resid(T, mask)
    dni_r3  = partial_resid(DNI, mask)
    dif_r3  = partial_resid(Dif, mask)
    qc_r3   = partial_resid(df['Electric cooling W'].values/1e6, mask)

    for ch_name, y_r, forcing_name, x_r in [
        ('Q_C   ← T_amb',  qc_r3, 'T_amb', tamb_r3),
        ('Q_C   ← DNI',    qc_r3, 'DNI',   dni_r3),
        ('Q_C   ← Diffuse',qc_r3, 'Diffuse',dif_r3),
    ]:
        coef, r2 = fit_fir(x_r, y_r)
        ds = delay_stats(coef)
        fir_results[('S3', ch_name)] = (coef, r2, ds)
        print(f"  {ch_name:22s}: R²={r2:.3f}  sign={ds['sign']:+.0f}  "
              f"peak={ds['peak']:3d}h  τ_eff={ds['tau']:4.1f}h  "
              f"50%={ds['lag50']:3d}h  90%={ds['lag90']:3d}h")
else:
    print(f"  Too few hours in State 3 ({mask.sum()}) for reliable FIR fit")

# ── Cross-state comparison: does τ change between states? ─────────────────────
print("\n" + "="*65)
print("DELAY COMPARISON: State 1 vs State 2 for Q_H ← T_wc")
print("="*65)
for skey in ['S1', 'S2']:
    key = (skey, 'Q_H   ← T_wc')
    if key in fir_results:
        _, r2, ds = fir_results[key]
        print(f"  {skey}: peak={ds['peak']}h  τ_eff={ds['tau']:.1f}h  "
              f"50%={ds['lag50']}h  90%={ds['lag90']}h  R²={r2:.3f}")

# ── FIGURES ────────────────────────────────────────────────────────────────────
# Figure 1: Impulse responses for key couplings
fig, axes = plt.subplots(3, 3, figsize=(16, 13))
fig.suptitle('Estimated Impulse Responses per Thermal State\n'
             'FIR coefficients = true delay distribution after removing diurnal + port/sea',
             fontsize=11, fontweight='bold')

STATE_COL = {'S1':'#4575b4', 'S2':'#313695', 'S3':'#d73027'}
STATE_LABEL = {'S1':'State 1 (cabin heating 5–21°C)',
               'S2':'State 2 (full heating <5°C)',
               'S3':'State 3 (cooling >24°C)'}

# Panel layout: rows=channels, cols=forcing
panels = [
    ('S1', 'Q_H   ← T_wc',    'ax00', 'Heating ← T_wc  [State 1]'),
    ('S2', 'Q_H   ← T_wc',    'ax01', 'Heating ← T_wc  [State 2]'),
    ('S3', 'Q_C   ← T_amb',   'ax02', 'Cooling ← T_amb [State 3]'),
    ('S1', 'Q_H   ← DNI',     'ax10', 'Heating ← DNI   [State 1]'),
    ('S3', 'Q_C   ← DNI',     'ax11', 'Cooling ← DNI   [State 3]'),
    ('S1', 'Q_H   ← Diffuse', 'ax12', 'Heating ← Diffuse [State 1]'),
    ('S1', 'HVAC  ← T_wc',    'ax20', 'HVAC ← T_wc  [State 1]'),
    ('S2', 'HVAC  ← T_wc',    'ax21', 'HVAC ← T_wc  [State 2]'),
    ('S1', 'DHW   ← T_wc',    'ax22', 'DHW ← T_wc   [State 1]'),
]

for (skey, ch, _, title), ax in zip(panels, axes.flat):
    key = (skey, ch)
    if key not in fir_results:
        ax.text(0.5,0.5,'insufficient\ndata', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')
        ax.set_title(title, fontsize=8.5); continue

    coef, r2, ds = fir_results[key]
    col = STATE_COL[skey]

    # Plot impulse response
    ax.fill_between(LAGS, 0, coef, alpha=0.35, color=col)
    ax.plot(LAGS, coef, color=col, lw=2)
    ax.axhline(0, color='k', lw=0.8, ls=':')

    # Mark key delays
    ax.axvline(ds['peak'], color='red',    lw=2, ls='--',
               label=f"peak={ds['peak']}h")
    ax.axvline(ds['tau'],  color='purple', lw=1.5, ls='-.',
               label=f"τ_eff={ds['tau']:.1f}h")
    ax.axvline(ds['lag90'],color='orange', lw=1.5, ls=':',
               label=f"90%={ds['lag90']}h")

    ax.set_title(f'{title}\nR²={r2:.3f}', fontsize=8.5, fontweight='bold')
    ax.set_xlabel('Lag (hours)'); ax.set_ylabel('β (MW per unit forcing)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    ax.set_xlim(0, MAX_LAG)

plt.tight_layout()
fig.savefig(OUTDIR+'cf_impulse_responses.png', dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved cf_impulse_responses.png")

# Figure 2: Summary delay table
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

rows = []
for (skey, ch_label), (coef, r2, ds) in sorted(fir_results.items()):
    state_str = STATE_LABEL[skey].split('(')[1].rstrip(')')
    rows.append([
        skey,
        ch_label.strip(),
        state_str,
        f"{ds['sign']:+.0f}",
        f"{ds['peak']}h",
        f"{ds['tau']:.1f}h",
        f"{ds['lag50']}h",
        f"{ds['lag90']}h",
        f"{r2:.3f}"
    ])

tbl = ax.table(
    cellText=rows,
    colLabels=['State','Coupling','Range','Sign','Peak lag',
               'τ_eff','50% lag','90% lag','R²'],
    cellLoc='center', loc='center',
    colWidths=[0.06,0.18,0.17,0.05,0.08,0.07,0.07,0.07,0.06])
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.7)

for (r,c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2166ac')
        cell.set_text_props(color='w', weight='bold')
    elif r > 0:
        skey = rows[r-1][0]
        cell.set_facecolor({'S1':'#dbeeff','S2':'#c5d8f5','S3':'#ffe0d8'}[skey])

ax.set_title('Impulse Response Delay Summary per Orthogonal State\n'
             'FIR fit after removing diurnal + port/sea confounders',
             fontsize=10, fontweight='bold', pad=20)
plt.tight_layout()
fig.savefig(OUTDIR+'cf_delay_table.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved cf_delay_table.png")
