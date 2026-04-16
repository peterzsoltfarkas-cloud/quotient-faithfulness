"""
Color Fantasy — complete physics analysis incorporating Peter's domain knowledge.

Produces:
  1. Corrected correlation interpretation table
  2. Discrete thermal state identification from data
  3. State-conditional manifold structure
  4. Revised input space (5D with windchill)
  5. Generator proposal document
"""
import numpy as np, pandas as pd, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

OUTDIR = '/mnt/user-data/outputs/figures/'

# ── Load data ──────────────────────────────────────────────────────────────────
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

T   = df['Air temp [°C]'].values
V   = df['Total wind [m/s]'].values          # m/s
DNI = df['Direct normal radiation [W/m2]'].values
Dif = df['Diffuse radition on horizontal surface [W/m2]'].values
RH  = df['Relative humidity [%]'].values
CC  = df['Cloud cover [%]'].values

Q_H    = df['Acc. heating W'].values / 1e6
Q_C    = df['Electric cooling W'].values / 1e6
Q_DHW  = df['Domestic hot water W'].values / 1e6
Q_G    = df['Galley steam demand W'].values / 1e6
Q_HVAC = df['HVAC aux W'].values / 1e6
Q_L    = df['Lighting facility W'].values / 1e6
Q_E    = df['Equipment tenant W'].values / 1e6

# ── Windchill effective temperature ───────────────────────────────────────────
# JAG/TI formula (Wikipedia Wind chill):
# T_wc = 13.12 + 0.6215*T - 11.37*V^0.16 + 0.3965*T*V^0.16  (V in km/h)
V_kmh = V * 3.6
T_wc  = np.where(
    T <= 10,  # windchill only applies below 10°C
    13.12 + 0.6215*T - 11.37*(V_kmh**0.16) + 0.3965*T*(V_kmh**0.16),
    T)
df['T_windchill'] = T_wc
df['T_amb']       = T

print("Windchill effect sample:")
for t, v, twc in zip([0,-5,5,10],[5,10,15,20],
                     [T_wc[(np.abs(T-t)<0.5)&(np.abs(V_kmh-v*3.6/3.6)<1)].mean()
                      for t,v in [(0,5),(-5,10),(5,15),(10,20)]]):
    print(f"  T={t}°C, V={v}m/s → T_wc≈{twc:.1f}°C")

# ── Identify thermal control states ───────────────────────────────────────────
# State 1: Heating – cabin only (5°C < T_wc ≤ 21°C)
# State 2: Heating – cabin + car deck (T_wc ≤ 5°C)
# State 3: Cooling  (T > 24°C)
# State 0: Transition / mild (21°C < T ≤ 24°C)

state = np.where(T_wc <= 5,   2,       # cold: double heating
        np.where(T_wc <= 21,  1,       # mild: cabin heating
        np.where(T   >= 24,   3,       # hot: cooling
                              0)))     # transition band 21–24°C

state_names = {0:'Transition\n(21–24°C)',
               1:'Cabin heating\n(5–21°C)',
               2:'Full heating\n(<5°C)',
               3:'Cooling\n(>24°C)'}
state_colors= {0:'#aaaaaa', 1:'#4575b4', 2:'#313695', 3:'#d73027'}

df['state'] = state
counts = {s: (state==s).sum() for s in [0,1,2,3]}
print("\nThermal state distribution:")
for s,n in counts.items():
    pct = n/len(state)*100
    print(f"  State {s} ({state_names[s].replace(chr(10),' ')}): "
          f"{n} h  ({pct:.1f}%)")

# ── State-conditional correlations ────────────────────────────────────────────
# For each state: what drives Q_H and Q_HVAC?
print("\nState-conditional partial r(T_wc, Q_H):")
for s in [1,2,3]:
    mask = state == s
    if mask.sum() < 100: continue
    r = np.corrcoef(T_wc[mask], Q_H[mask])[0,1]
    r_t  = np.corrcoef(T[mask], Q_H[mask])[0,1]
    r_wc = np.corrcoef(T_wc[mask], Q_H[mask])[0,1]
    r_v  = np.corrcoef(V[mask], Q_H[mask])[0,1]
    print(f"  State {s} (n={mask.sum()}): "
          f"r(T,Q_H)={r_t:.3f}  r(T_wc,Q_H)={r_wc:.3f}  r(V,Q_H)={r_v:.3f}")

print("\nQ_HVAC per state (mean MW):")
for s in [0,1,2,3]:
    mask = state == s
    if mask.sum() < 100: continue
    print(f"  State {s}: Q_HVAC mean={Q_HVAC[mask].mean():.4f} MW  "
          f"std={Q_HVAC[mask].std():.4f}")

# ── Radiation independence ─────────────────────────────────────────────────────
# Verify DNI and Diffuse are independent drivers of cooling
print("\nCooling: radiation contributions (state 3 only, T>24°C):")
mask3 = state == 3
if mask3.sum() > 50:
    # Multiple regression: Q_C ~ T + DNI + Diffuse
    from sklearn.linear_model import LinearRegression
    Xr = np.column_stack([T[mask3], DNI[mask3], Dif[mask3]])
    lr = LinearRegression().fit(Xr, Q_C[mask3])
    print(f"  Coefficients: β_T={lr.coef_[0]:.4f}  "
          f"β_DNI={lr.coef_[1]:.6f}  β_Diffuse={lr.coef_[2]:.6f}")
    print(f"  R² = {lr.score(Xr, Q_C[mask3]):.3f}")
    # Each as standalone
    for name, x in [('T_amb',T[mask3]),('DNI',DNI[mask3]),('Diffuse',Dif[mask3])]:
        r = np.corrcoef(x, Q_C[mask3])[0,1]
        print(f"  r({name}, Q_C | state=3) = {r:.3f}")


# ── FIGURES ────────────────────────────────────────────────────────────────────

# Figure 1: Thermal state identification + windchill
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Color Fantasy — Thermal Control States and Windchill Effect\n'
             'Hard switching between 4 states creates orthogonal sub-manifolds',
             fontsize=11, fontweight='bold')

# Panel 1: T vs T_windchill scatter
ax = axes[0,0]
sc = ax.scatter(T[::4], T_wc[::4], c=state[::4],
                cmap=plt.cm.RdYlBu_r, s=3, alpha=0.4, vmin=0, vmax=3)
ax.plot([-15,35],[-15,35],'k--',lw=1,alpha=0.5,label='T_wc = T')
ax.axhline(5,  color='#4575b4', lw=2, ls='--', label='5°C car-deck threshold')
ax.axhline(21, color='#313695', lw=2, ls='--', label='21°C heating off')
ax.axhline(24, color='#d73027', lw=2, ls='--', label='24°C cooling on')
ax.set_xlabel('T_amb (°C)'); ax.set_ylabel('T_windchill (°C)')
ax.set_title('Windchill effective temperature\n'
             'Diverges from T_amb at low T and high wind', fontweight='bold')
ax.legend(fontsize=7.5); ax.grid(True,alpha=0.2)
plt.colorbar(sc,ax=ax,label='Thermal state')

# Panel 2: State distribution over year
ax = axes[0,1]
day_arr = df['day'].values
for s in [1,2,3,0]:
    mask = state == s
    # daily fraction in this state
    state_daily = np.array([
        (state[day_arr==d] == s).mean()
        for d in range(366) if (day_arr==d).sum()>0
    ])
    days_with_data = sorted(set(day_arr))[:len(state_daily)]
    ax.fill_between(days_with_data, state_daily, alpha=0.6,
                    color=state_colors[s], label=state_names[s].replace('\n',' '))
ax.set_xlabel('Day of year'); ax.set_ylabel('Fraction of hours in state')
ax.set_title('Thermal state occupancy through the year\n'
             'States are discrete, not gradual', fontweight='bold')
ax.legend(fontsize=7.5,ncol=2); ax.grid(True,alpha=0.2)
for month, label in zip([0,31,59,90,120,151,181,212,243,273,304,334],
                         ['Jan','Feb','Mar','Apr','May','Jun',
                          'Jul','Aug','Sep','Oct','Nov','Dec']):
    ax.axvline(month, color='gray', lw=0.5, alpha=0.4)
    ax.text(month+2, 1.02, label, fontsize=7, va='bottom', color='gray')

# Panel 3: Q_H vs T_wc per state
ax = axes[1,0]
for s in [1,2]:
    mask = state == s
    ax.scatter(T_wc[mask][::4], Q_H[mask][::4],
               s=3, alpha=0.3, color=state_colors[s],
               label=state_names[s].replace('\n',' '))
    # Fit line
    lr = LinearRegression().fit(T_wc[mask].reshape(-1,1), Q_H[mask])
    t_range = np.linspace(T_wc[mask].min(), T_wc[mask].max(), 100)
    ax.plot(t_range, lr.predict(t_range.reshape(-1,1)),
            color=state_colors[s], lw=2.5,
            label=f'State {s}: slope={lr.coef_[0]:.3f} MW/°C')
ax.axvline(5, color='k', lw=1.5, ls='--', alpha=0.5, label='5°C switch')
ax.set_xlabel('T_windchill (°C)'); ax.set_ylabel('Q_Heating (MW)')
ax.set_title('Heating demand vs windchill\n'
             'Two distinct linear regimes with hard switch at 5°C', fontweight='bold')
ax.legend(fontsize=7.5); ax.grid(True,alpha=0.2)

# Panel 4: HVAC per state — shows step change at state 2
ax = axes[1,1]
for s in [0,1,2,3]:
    mask = state == s
    if mask.sum() < 50: continue
    ax.violinplot([Q_HVAC[mask]], positions=[s], widths=0.7,
                  showmedians=True)
hvac_means = [Q_HVAC[state==s].mean() for s in [0,1,2,3] if (state==s).sum()>50]
states_with_data = [s for s in [0,1,2,3] if (state==s).sum()>50]
ax.scatter(states_with_data, hvac_means, s=80, color='red', zorder=5, label='Mean')
ax.set_xticks([0,1,2,3])
ax.set_xticklabels([state_names[s].replace('\n','\n') for s in [0,1,2,3]],
                    fontsize=8.5)
ax.set_ylabel('Q_HVAC (MW)')
ax.set_title('HVAC by thermal state\n'
             'Step increase at state 2: car deck ventilation added', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True,alpha=0.2,axis='y')
# Annotate the jump
ax.annotate('Car deck\nadded', xy=(2, Q_HVAC[state==2].mean()),
            xytext=(2.3, Q_HVAC[state==2].mean()+0.05),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=8.5, color='red')

plt.tight_layout()
fig.savefig(OUTDIR+'cf_thermal_states.png', dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved cf_thermal_states.png")

# Figure 2: State-space manifold structure (input space coloured by state)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Input Space Coloured by Thermal State\n'
             'Hard switches create orthogonal sub-manifolds — NOT a smooth quotient',
             fontsize=11, fontweight='bold')

# T_wc vs DNI
ax = axes[0]
for s in [1,2,3,0]:
    mask = state == s
    if mask.sum() < 50:
        continue
    ax.scatter(T_wc[mask][::4], DNI[mask][::4],
               s=4, alpha=0.3, color=state_colors[s],
               label=state_names[s].replace('\n',' '))
ax.axvline(5,  color='k', lw=1.5, ls='--', alpha=0.6)
ax.axvline(21, color='k', lw=1.5, ls='--', alpha=0.6)
ax.set_xlabel('T_windchill (°C)'); ax.set_ylabel('DNI (W/m²)')
ax.set_title('T_windchill × DNI space\nVertical bands = thermal states')
ax.legend(fontsize=7.5, markerscale=3); ax.grid(True,alpha=0.2)

# T_wc vs Diffuse
ax = axes[1]
for s in [1,2,3,0]:
    mask = state == s
    if mask.sum() < 50: continue
    ax.scatter(T_wc[mask][::4], Dif[mask][::4],
               s=4, alpha=0.3, color=state_colors[s],
               label=state_names[s].replace('\n',' '))
ax.axvline(5,  color='k', lw=1.5, ls='--', alpha=0.6)
ax.axvline(21, color='k', lw=1.5, ls='--', alpha=0.6)
ax.set_xlabel('T_windchill (°C)'); ax.set_ylabel('Diffuse (W/m²)')
ax.set_title('T_windchill × Diffuse space')
ax.legend(fontsize=7.5, markerscale=3); ax.grid(True,alpha=0.2)

# DNI vs Diffuse
ax = axes[2]
for s in [1,2,3,0]:
    mask = state == s
    if mask.sum() < 50: continue
    ax.scatter(DNI[mask][::4], Dif[mask][::4],
               s=4, alpha=0.3, color=state_colors[s],
               label=state_names[s].replace('\n',' '))
ax.set_xlabel('DNI (W/m²)'); ax.set_ylabel('Diffuse (W/m²)')
ax.set_title('DNI × Diffuse space\nIndependent radiation coordinates')
ax.legend(fontsize=7.5, markerscale=3); ax.grid(True,alpha=0.2)

plt.tight_layout()
fig.savefig(OUTDIR+'cf_state_manifolds.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved cf_state_manifolds.png")
