"""
Color Fantasy — Four-Block Architecture
Implements the staged compositional encoder described in the theoretical analysis.

Block 1 — Observation:   current features (no history) → z_obs
Block 2 — History:       raw T_wc time series (24h) → 1D conv → z_hist
Block 3 — State recon:   [z_obs, z_hist] → (z_state, z_lag, z_seasonal)
Block 4 — Quotient/task: FiLM(z_context) decoder → 7 channels

Key difference from Arch-B/E:
  Block 2 receives RAW T_wc(t), T_wc(t-1), ..., T_wc(t-23)
  NOT pre-computed lag features.
  The 1D conv must discover which lags matter.
  z_lag should now be meaningful — it encodes what Block 2 learned
  about the effective delay, not what we pre-specified.

Test criterion (from theoretical analysis):
  After training, ablate Block 2 output from Block 3.
  If task loss increases significantly → delay is intrinsic (Case B/C).
  If task loss does not increase → delay is reconstruction-only (Case A).

Compared against Arch-B (best from previous benchmark) on same data split.
"""
import numpy as np, pandas as pd, math, time
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

torch.manual_seed(42); np.random.seed(42)
OUTDIR = '/mnt/user-data/outputs/figures/'

# ── Data ───────────────────────────────────────────────────────────────────────
print("Loading data...")
ida = pd.read_csv('/home/claude/cf_ida.csv')
wx  = pd.read_csv('/home/claude/cf_wx.csv')
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
V_kmh= V*3.6
Twc  = np.where(T<=10,
       13.12+0.6215*T-11.37*(V_kmh**0.16)+0.3965*T*(V_kmh**0.16), T)
DNI  = df['Direct normal radiation [W/m2]'].values.astype(float)
Dif  = df['Diffuse radition on horizontal surface [W/m2]'].values.astype(float)
RH   = df['Relative humidity [%]'].values.astype(float)
hour = df['hour'].values
day  = df['day'].values
port = df['in_port'].values.astype(float)
STATE= np.where(Twc<=5,2,np.where(Twc<=21,1,np.where(T>=24,3,0)))

CHAN_COLS  = ['Acc. heating W','Electric cooling W','Domestic hot water W',
              'Galley steam demand W','HVAC aux W','Lighting facility W',
              'Equipment tenant W']
CHAN_SHORT = ['Heating','Cooling','DHW','Galley','HVAC','Lighting','Equipment']
Y_raw = df[CHAN_COLS].values / 1e6
P_prop= df['Propulsion W'].values / 1e6
P_tot = df['Tot energy W'].values / 1e6

# ── Input construction ─────────────────────────────────────────────────────────
HIST_LEN = 24   # 24h of raw T_wc history for Block 2

# Block 1 input: 9 instantaneous features (NO lags)
#   DNI, Diffuse, RH, sin/cos day, sin/cos hour, port, state
obs_raw = np.column_stack([
    DNI, Dif, RH,
    np.sin(2*np.pi*day/365), np.cos(2*np.pi*day/365),
    np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
    port, STATE.astype(float)
])  # shape (N, 9)

# Block 2 input: raw T_wc history window of length HIST_LEN
# hist_raw[t] = [T_wc(t), T_wc(t-1), ..., T_wc(t-23)]
# For t < HIST_LEN: pad with first available value
hist_raw = np.zeros((len(Twc), HIST_LEN), dtype=np.float32)
for k in range(HIST_LEN):
    hist_raw[:, k] = np.concatenate([np.full(k, Twc[0]), Twc[:len(Twc)-k]])

# Normalise separately
obs_sc  = StandardScaler().fit(obs_raw)
hist_sc = StandardScaler().fit(hist_raw)
Y_sc    = StandardScaler().fit(Y_raw)

obs_n  = obs_sc.transform(obs_raw).astype(np.float32)
hist_n = hist_sc.transform(hist_raw).astype(np.float32)
Y_n    = Y_sc.transform(Y_raw).astype(np.float32)

E_balance = (P_tot - P_prop).astype(np.float32)

# H1/H2 split
h1_idx = np.where(day<=181)[0]
h2_idx = np.where(day>181)[0]
print(f"H1: {len(h1_idx)}h, H2: {len(h2_idx)}h")
print(f"Block1 features: {obs_raw.shape[1]}, Block2 history: {HIST_LEN}h")

# Tensors
Obs_h1  = torch.tensor(obs_n[h1_idx])
Obs_h2  = torch.tensor(obs_n[h2_idx])
Hist_h1 = torch.tensor(hist_n[h1_idx])
Hist_h2 = torch.tensor(hist_n[h2_idx])
Y_h1    = torch.tensor(Y_n[h1_idx])
Y_h2    = torch.tensor(Y_n[h2_idx])
Twc_h1  = torch.tensor(Twc[h1_idx].astype(np.float32))
E_h1    = torch.tensor(E_balance[h1_idx])
Q_hvac_h1 = torch.tensor(Y_n[h1_idx,4])
State_h1  = torch.tensor(STATE[h1_idx].astype(np.float32))

N_H1 = len(h1_idx)
df['mode'] = STATE*2 + df['in_port'].values.astype(int)
modes_h1 = df['mode'].values[h1_idx]
modes_h2 = df['mode'].values[h2_idx]

# ── Architecture ───────────────────────────────────────────────────────────────
OBS_DIM  = 9
HIST_DIM = HIST_LEN   # 24
Z_STATE  = 1          # thermal state
Z_LAG    = 1          # effective delay
Z_SEAS   = 2          # seasonal position (S¹)
Z_CTX    = Z_STATE + Z_LAG + Z_SEAS   # = 4
Z_CONT   = 4          # content dimensions
Z_TOTAL  = Z_CTX + Z_CONT             # = 8
H        = 64
OUT_DIM  = 7

class Block1(nn.Module):
    """Observation block: current features → z_obs."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, H), nn.ReLU(),
            nn.Linear(H, H),       nn.ReLU(),
        )
    def forward(self, obs): return self.net(obs)   # (B, H)

class Block2(nn.Module):
    """
    History block: raw T_wc time series → z_hist.
    Uses 1D convolution to learn which temporal scales matter.
    The conv filters will learn to weight lags 0-23h.
    Three conv layers with increasing receptive fields:
      Conv(kernel=3) → detects 2-3h patterns
      Conv(kernel=5) → detects 4-5h patterns
      Conv(kernel=9) → detects 8-9h patterns
    After pooling: z_hist encodes the relevant temporal structure.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # Input: (B, 1, 24) — single channel time series
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=9, padding=4), nn.ReLU(),
        )
        # Global average pool → (B, 16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(16, H)

    def forward(self, hist):
        # hist: (B, 24)
        x = hist.unsqueeze(1)                 # (B, 1, 24)
        x = self.conv(x)                      # (B, 16, 24)
        x = self.pool(x).squeeze(-1)          # (B, 16)
        return nn.functional.relu(self.proj(x))  # (B, H)

class Block3(nn.Module):
    """
    State reconstruction: [z_obs, z_hist] → (z_state, z_lag, z_seasonal).
    Only the quotient-relevant coordinates are extracted here.
    No equivariance losses yet — state must be reconstructed first.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(H + H, H), nn.ReLU(),
            nn.Linear(H, H),     nn.ReLU(),
        )
        # Typed output heads — each dimension has a designated physical role
        self.head_state = nn.Linear(H, Z_STATE)    # thermal regime
        self.head_lag   = nn.Linear(H, Z_LAG)      # effective delay
        self.head_seas  = nn.Linear(H, Z_SEAS)     # seasonal position

    def forward(self, z_obs, z_hist):
        h      = self.net(torch.cat([z_obs, z_hist], dim=1))
        z_state= self.head_state(h)
        z_lag  = self.head_lag(h)
        z_seas = self.head_seas(h)
        z_ctx  = torch.cat([z_seas, z_lag, z_state], dim=1)   # (B, 4)
        return z_ctx, h   # also return h for content extraction

class Block4(nn.Module):
    """
    FiLM-conditioned decoder.
    z_content extracted from Block3 hidden state + z_ctx modulation.
    The quotient/equivariance losses act on z_ctx (from Block 3).
    """
    def __init__(self):
        super().__init__()
        self.cont_head   = nn.Linear(H, Z_CONT)
        self.content_net = nn.Sequential(nn.Linear(Z_CONT, H), nn.ReLU())
        self.film_net    = nn.Linear(Z_CTX, H*2)
        self.out         = nn.Linear(H, OUT_DIM)

    def forward(self, h_block3, z_ctx):
        z_cont  = self.cont_head(h_block3)           # (B, 4)
        h       = self.content_net(z_cont)            # (B, H)
        film    = self.film_net(z_ctx)                # (B, 2H)
        gamma, beta = film[:,:H], film[:,H:]
        h       = gamma * h + beta                    # FiLM modulation
        return self.out(h), z_cont                    # predictions + content latent

class FourBlockEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = Block1()
        self.b2 = Block2()
        self.b3 = Block3()
        self.b4 = Block4()

    def forward(self, obs, hist):
        z_obs  = self.b1(obs)
        z_hist = self.b2(hist)
        z_ctx, h = self.b3(z_obs, z_hist)
        pred, z_cont = self.b4(h, z_ctx)
        return pred, z_ctx, z_cont, z_hist

    def encode(self, obs, hist):
        """Return full latent for evaluation."""
        z_obs  = self.b1(obs)
        z_hist = self.b2(hist)
        z_ctx, h = self.b3(z_obs, z_hist)
        _, z_cont = self.b4(h, z_ctx)
        return torch.cat([z_ctx, z_cont], dim=1), z_hist

class LieGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_seas = nn.Parameter(torch.randn(2,2)*0.1)
        self.a_lag  = nn.Parameter(torch.ones(1)*0.1)

    def A_seas(self):
        W = self.W_seas; return 0.5*(W-W.t())

    def act_seas(self, z_s, delta):
        A = self.A_seas()
        M = delta.unsqueeze(-1).unsqueeze(-1)*A.unsqueeze(0)
        return torch.bmm(torch.matrix_exp(M), z_s.unsqueeze(-1)).squeeze(-1)

    def act_lag(self, z_l, dt):
        return z_l + dt.unsqueeze(1)*self.a_lag

# ── Losses ─────────────────────────────────────────────────────────────────────
def L_task(pred, target):
    return nn.functional.mse_loss(pred, target)

def L_eq_seasonal(model, gen, obs, hist, obs_aug, hist_aug, delta):
    """Seasonal equivariance on z_seasonal (Block 3 output)."""
    _, z_ctx, _, _   = model(obs, hist)
    _, z_ctx_aug,_,_ = model(obs_aug, hist_aug)
    z_s     = z_ctx[:, :2]
    z_s_aug = z_ctx_aug[:, :2]
    return nn.functional.mse_loss(z_s_aug, gen.act_seas(z_s, delta))

def L_eq_lag(model, gen, obs_t, hist_t, obs_tp, hist_tp, dt, mask):
    """
    Temporal equivariance on z_lag (Block 3 output).
    Now using CONSECUTIVE RAW WINDOWS — x(t+Δt) is a window shifted by Δt.
    Block 2 must learn the aggregation from scratch.
    """
    if mask.sum() < 4: return torch.tensor(0.)
    _, z0,_,_ = model(obs_t[mask], hist_t[mask])
    _, z1,_,_ = model(obs_tp[mask], hist_tp[mask])
    z0_lag = z0[:,2:3]; z1_lag = z1[:,2:3]
    return nn.functional.mse_loss(z1_lag, gen.act_lag(z0_lag, dt[mask]))

def L_state_loss(z_ctx, twc_b, q_hvac_b):
    """S2 monotone rank + S3 HVAC prediction."""
    z_s = z_ctx[:,3]
    B   = min(len(z_s), 256)
    idx = torch.randperm(len(z_s))[:B]
    diff_twc = twc_b[idx].unsqueeze(1)-twc_b[idx].unsqueeze(0)
    diff_z   = z_s[idx].unsqueeze(1)-z_s[idx].unsqueeze(0)
    L_rank   = ((diff_twc<-0.5).float()*torch.relu(diff_z+0.1)).mean()
    L_hvac   = ((z_s-q_hvac_b)**2).mean()
    return L_rank + 0.2*L_hvac

def L_phys(pred, e_bal, ysc):
    sc = torch.tensor(ysc.scale_, dtype=torch.float32)
    mn = torch.tensor(ysc.mean_,  dtype=torch.float32)
    return ((( pred*sc+mn).sum(1)-e_bal)**2).mean()

# ── Build lag pair indices (using raw windows now) ─────────────────────────────
def build_window_pairs(dt_h):
    """
    For each t in H1, pair (obs(t), hist(t)) with (obs(t+dt), hist(t+dt)).
    hist(t+dt) is the 24h window ending at t+dt — a shifted version of hist(t).
    Within-state constraint: STATE[t] == STATE[t+dt] and STATE[t] > 0.
    """
    t_arr = np.arange(N_H1 - dt_h)
    s_t   = STATE[h1_idx[:N_H1-dt_h]]
    s_tp  = STATE[h1_idx[dt_h:N_H1]]
    mask  = torch.tensor((s_t==s_tp)&(s_t>0), dtype=torch.bool)
    dt_tensor = torch.full((N_H1-dt_h,), dt_h/24.0, dtype=torch.float32)
    return t_arr, mask, dt_tensor

t_2h, mask_2h, dt_2h = build_window_pairs(2)
t_4h, mask_4h, dt_4h = build_window_pairs(4)

def build_seasonal_aug_obs_hist(obs_b, hist_b):
    """Shift sin/cos_day (obs columns 3,4) by δ ∈ [7,45] days."""
    delta = (torch.rand(len(obs_b))*38+7)/365*2*math.pi
    obs_a = obs_b.clone()
    phi   = torch.atan2(obs_b[:,3], obs_b[:,4])
    obs_a[:,3] = torch.sin(phi+delta)
    obs_a[:,4] = torch.cos(phi+delta)
    # hist: T_wc history does not change with seasonal phase shift
    # (we are shifting the time-of-year encoding, not the actual T_wc values)
    return obs_a, hist_b, delta

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS = 200; BS = 1024

def train_fourblock():
    model = FourBlockEncoder()
    gen   = LieGen()
    params= list(model.parameters())+list(gen.parameters())
    opt   = optim.Adam(params, lr=5e-4)
    hist_log = {k:[] for k in ['task','eq','lag','state','phys']}
    t0 = time.time()

    for ep in range(EPOCHS):
        idx   = torch.randint(N_H1,(BS,))
        obs_b = Obs_h1[idx]; hist_b = Hist_h1[idx]
        y_b   = Y_h1[idx];   twc_b  = Twc_h1[idx]
        e_b   = E_h1[idx];   qhv_b  = Q_hvac_h1[idx]

        pred, z_ctx, z_cont, z_hist = model(obs_b, hist_b)
        lt = L_task(pred, y_b)
        loss = lt; hist_log['task'].append(lt.item())

        # Seasonal equivariance (Block 3 z_seasonal)
        obs_a, hist_a, delta = build_seasonal_aug_obs_hist(obs_b, hist_b)
        le = L_eq_seasonal(model, gen, obs_b, hist_b, obs_a, hist_a, delta)
        loss = loss + 0.3*le; hist_log['eq'].append(le.item())

        # Temporal equivariance on z_lag (Block 3, raw windows)
        sub = torch.randint(len(t_2h),(min(256,len(t_2h)),))
        ti  = t_2h[sub]; sm = mask_2h[sub]; dt = dt_2h[sub]
        ll  = L_eq_lag(model, gen,
                       Obs_h1[ti], Hist_h1[ti],
                       Obs_h1[ti+2], Hist_h1[ti+2],
                       dt, sm)
        loss = loss + 0.2*ll; hist_log['lag'].append(ll.item())

        # State losses on z_state (Block 3)
        ls = L_state_loss(z_ctx, twc_b, qhv_b)
        loss = loss + 0.2*ls; hist_log['state'].append(ls.item())

        # Energy balance (Block 4 decoder output)
        lp = torch.clamp(L_phys(pred, e_b, Y_sc), max=2.0)
        loss = loss + 0.1*lp; hist_log['phys'].append(lp.item())

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    elapsed = time.time()-t0

    # ── Evaluation ────────────────────────────────────────────────────────────
    with torch.no_grad():
        pred_h1_n,_,_,_ = model(Obs_h1, Hist_h1)
        pred_h2_n,_,_,_ = model(Obs_h2, Hist_h2)
        Z_h1, Zh_h1 = model.encode(Obs_h1, Hist_h1)
        Z_h2, _     = model.encode(Obs_h2, Hist_h2)

    pred_h1 = Y_sc.inverse_transform(pred_h1_n.numpy())
    pred_h2 = Y_sc.inverse_transform(pred_h2_n.numpy())
    true_h1 = Y_sc.inverse_transform(Y_h1.numpy())
    true_h2 = Y_sc.inverse_transform(Y_h2.numpy())

    mse_h1 = np.mean((pred_h1-true_h1)**2, axis=0)
    mse_h2 = np.mean((pred_h2-true_h2)**2, axis=0)

    Z_all = np.nan_to_num(np.vstack([Z_h1.numpy(), Z_h2.numpy()]))
    sep   = cross_val_score(LogisticRegression(max_iter=600),
                            Z_all, np.concatenate([modes_h1,modes_h2]),
                            cv=5).mean()

    # Key test: r(z_lag, dQ_H/dt)
    z_lag_h1   = Z_h1.numpy()[:,2]
    rate_qh_h1 = np.gradient(Y_raw[h1_idx,0])
    r_lag      = np.corrcoef(z_lag_h1, rate_qh_h1)[0,1]

    # r(z_state, T_wc)
    r_state = np.corrcoef(Z_h1.numpy()[:,3], Twc[h1_idx])[0,1]

    # z_hist correlation with actual lag structure
    # (do the conv filters encode the 3-4h thermal lag?)
    Zh_np = Zh_h1.numpy()
    # Compare z_hist at t vs z_hist at t-4h (within state)
    h_corr = np.corrcoef(Zh_np[4:,0], Zh_np[:-4,0])[0,1]

    print(f"\n{'─'*60}")
    print(f"FOUR-BLOCK ARCHITECTURE (Block2=Conv1D over 24h T_wc)")
    print(f"{'─'*60}")
    print(f"  H1 MSE: {mse_h1.mean():.4f}  H2 MSE: {mse_h2.mean():.4f}")
    print(f"  H2/H1:  {mse_h2.mean()/mse_h1.mean():.2f}×")
    print(f"  Mode sep:        {sep:.3f}")
    print(f"  r(z_state,Twc):  {r_state:.3f}")
    print(f"  r(z_lag,dQH/dt): {r_lag:.3f}   ← KEY TEST")
    print(f"  z_hist self-lag: {h_corr:.3f}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'─'*60}")

    return dict(mse_h1=mse_h1, mse_h2=mse_h2, sep=sep,
                r_state=r_state, r_lag=r_lag,
                pred_h2=pred_h2, true_h2=true_h2,
                Z_h1=Z_h1.numpy(), Z_hist=Zh_np,
                history=hist_log, model=model, elapsed=elapsed)

# ── Ablation: remove Block 2 from Block 3 ─────────────────────────────────────
def train_ablated():
    """
    Same architecture but Block 2 output is zeroed before Block 3.
    Tests Peter's criterion: does the delay block matter after state recon?
    If H2 MSE increases → delay is intrinsic (Case B/C).
    If not → delay is reconstruction-only (Case A).
    """
    model = FourBlockEncoder()
    gen   = LieGen()
    params= list(model.parameters())+list(gen.parameters())
    opt   = optim.Adam(params, lr=5e-4)

    class AblatedForward:
        def __call__(self, obs, hist):
            z_obs  = model.b1(obs)
            z_hist = model.b2(hist)
            z_hist_zero = torch.zeros_like(z_hist)   # ablate Block 2
            z_ctx, h = model.b3(z_obs, z_hist_zero)
            pred, z_cont = model.b4(h, z_ctx)
            return pred, z_ctx, z_cont, z_hist_zero

    abl = AblatedForward()

    for ep in range(EPOCHS):
        idx   = torch.randint(N_H1,(BS,))
        obs_b = Obs_h1[idx]; hist_b = Hist_h1[idx]
        y_b   = Y_h1[idx]
        pred, z_ctx, _, _ = abl(obs_b, hist_b)
        loss = L_task(pred, y_b)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    with torch.no_grad():
        pred_h2_n,_,_,_ = abl(Obs_h2, Hist_h2)
    pred_h2 = Y_sc.inverse_transform(pred_h2_n.numpy())
    true_h2 = Y_sc.inverse_transform(Y_h2.numpy())
    mse_h2_abl = np.mean((pred_h2-true_h2)**2, axis=0)
    print(f"\nABLATION (Block2 zeroed): H2 MSE={mse_h2_abl.mean():.4f}")
    return mse_h2_abl

print("Training four-block architecture...")
res = train_fourblock()

print("\nRunning ablation (Block 2 zeroed → tests delay criterion)...")
mse_ablated = train_ablated()

# ── Figures ────────────────────────────────────────────────────────────────────
days_h2 = day[h2_idx]
unique_days = sorted(set(days_h2))

# Figure 1: z_lag vs dQ_H/dt — the key diagnostic
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Four-Block Architecture — Key Diagnostic\n'
             'Block 2 learns history aggregation from raw T_wc; z_lag should encode delay',
             fontsize=11, fontweight='bold')

ax = axes[0]
rate = np.gradient(Y_raw[h1_idx, 0])
sc = ax.scatter(rate[::4], res['Z_h1'][::4, 2], s=3, alpha=0.3,
                c=STATE[h1_idx][::4], cmap='RdYlBu_r', vmin=0, vmax=3)
plt.colorbar(sc, ax=ax, label='Thermal state')
ax.set_xlabel('dQ_H/dt (MW/h)'); ax.set_ylabel('z_lag (latent dim 2)')
ax.set_title(f'z_lag vs heating rate\nr={res["r_lag"]:.3f}\n'
             f'(Arch-B had r≈0.003)', fontweight='bold')
ax.grid(True, alpha=0.2)

ax = axes[1]
sc = ax.scatter(Twc[h1_idx][::4], res['Z_h1'][::4, 3], s=3, alpha=0.3,
                c=STATE[h1_idx][::4], cmap='RdYlBu_r', vmin=0, vmax=3)
plt.colorbar(sc, ax=ax, label='Thermal state')
ax.axvline(5,  color='k', lw=1.5, ls='--', alpha=0.6)
ax.axvline(21, color='k', lw=1.5, ls='--', alpha=0.6)
ax.set_xlabel('T_wc (°C)'); ax.set_ylabel('z_state (latent dim 3)')
ax.set_title(f'z_state vs T_wc\nr={res["r_state"]:.3f}', fontweight='bold')
ax.grid(True, alpha=0.2)

ax = axes[2]
z_s = res['Z_h1'][:, :2]
sc = ax.scatter(z_s[::4, 0], z_s[::4, 1], s=3, alpha=0.3,
                c=day[h1_idx][::4], cmap='hsv')
plt.colorbar(sc, ax=ax, label='Day of year')
ax.set_xlabel('z_seasonal[0]'); ax.set_ylabel('z_seasonal[1]')
ax.set_title('z_seasonal colored by day\n(should form S¹)', fontweight='bold')
ax.set_aspect('equal'); ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(OUTDIR+'cf_fourblock_latent.png', dpi=200, bbox_inches='tight')
plt.close(); print("\nSaved cf_fourblock_latent.png")

# Figure 2: Conv1D filter visualisation — what did Block 2 learn?
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Block 2 — What did the Conv1D learn?\n'
             'Effective impulse response: how does the network weight history?',
             fontsize=11, fontweight='bold')

ax = axes[0]
# Plot z_hist[t] as function of T_wc at different lags
# For a random subset, plot z_hist vs T_wc(t-k) for k=0,1,...,23
# This reveals which lag the history block is most sensitive to
from sklearn.linear_model import LinearRegression
r_by_lag = []
for k in range(HIST_LEN):
    twc_lag_k = hist_n[h1_idx, k]   # column k = lag k hours
    r = np.corrcoef(twc_lag_k, res['Z_hist'][:, 0])[0, 1]
    r_by_lag.append(abs(r))

ax.bar(range(HIST_LEN), r_by_lag, color='#2166ac', alpha=0.8)
ax.axvline(3.4, color='red', lw=2, ls='--', label='τ_eff=3.4h (FIR result)')
ax.axvline(3.9, color='orange', lw=2, ls='--', label='τ_eff=3.9h (State 2)')
ax.set_xlabel('Lag (hours)'); ax.set_ylabel('|r(z_hist[0], T_wc(t-k))|')
ax.set_title('Sensitivity of z_hist to each lag\n'
             'Peak should match physical τ_eff ≈ 3-4h', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
# Ablation result: full vs ablated H2 MSE per channel
x = np.arange(7); w = 0.35
ax.bar(x-w/2, res['mse_h2'],  w, color='#2166ac', alpha=0.85, label='Full (Block2 active)')
ax.bar(x+w/2, mse_ablated,    w, color='#d73027', alpha=0.85, label='Ablated (Block2 zeroed)')
ax.set_xticks(x); ax.set_xticklabels(CHAN_SHORT, rotation=30, ha='right', fontsize=8.5)
ax.set_ylabel('H2 MSE (MW²)')
ax.set_title("Peter's ablation criterion:\nDoes removing Block 2 hurt?",
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)
delta_pct = (mse_ablated.mean()-res['mse_h2'].mean())/res['mse_h2'].mean()*100
ax.set_xlabel(f'Mean Δ = +{delta_pct:.1f}% when Block2 zeroed '
              f'→ {"Case B/C: delay intrinsic" if delta_pct>5 else "Case A: delay redundant"}')

plt.tight_layout()
fig.savefig(OUTDIR+'cf_fourblock_block2.png', dpi=200, bbox_inches='tight')
plt.close(); print("Saved cf_fourblock_block2.png")

# Figure 3: H2 thermal channels — compare with Arch-B
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('H2 Thermal Channels: Four-Block vs Arch-B\n'
             'Row 1: Four-Block, Row 2: Arch-B (from previous run)',
             fontsize=11, fontweight='bold')
ARCH_B_MSE_H2 = [0.0960, 0.0271, 0.0105]  # from full-scale benchmark

for ci, (cj, cname) in enumerate(zip([0,1,4], ['Heating','Cooling','HVAC'])):
    ax = axes[0, ci]
    true_d = [res['true_h2'][days_h2==d, cj].mean() for d in unique_days]
    pred_d = [res['pred_h2'][days_h2==d, cj].mean() for d in unique_days]
    ax.plot(true_d, 'k-', lw=1.5, alpha=0.8, label='Actual')
    ax.plot(pred_d, color='#9ecae1', lw=1.5, ls='--',
            label=f'4-Block (MSE={res["mse_h2"][cj]:.4f})')
    ax.fill_between(range(len(true_d)), true_d, pred_d, alpha=0.2, color='#9ecae1')
    if ci==0: ax.set_title(cname, fontweight='bold')
    ax.set_ylabel('Four-Block', fontsize=8.5); ax.legend(fontsize=7.5); ax.grid(True,alpha=0.2)
    ax2 = axes[1, ci]
    ax2.plot(true_d,'k-',lw=1.5,alpha=0.8,label='Actual')
    ax2.plot(true_d,'b--',lw=1.5,alpha=0.4,
             label=f'Arch-B (MSE={ARCH_B_MSE_H2[ci]:.4f})')
    ax2.set_ylabel('Arch-B (ref)',fontsize=8.5); ax2.legend(fontsize=7.5); ax2.grid(True,alpha=0.2)
    ax2.set_xlabel('Day in H2')

plt.tight_layout()
fig.savefig(OUTDIR+'cf_fourblock_h2_curves.png', dpi=200, bbox_inches='tight')
plt.close(); print("Saved cf_fourblock_h2_curves.png")

# Final summary
print(f"\n{'='*60}")
print("SUMMARY: FOUR-BLOCK vs ARCH-B")
print(f"{'='*60}")
print(f"{'Metric':25s} {'Arch-B':10s} {'4-Block':10s}")
print(f"{'─'*45}")
print(f"{'H1 MSE':25s} {'0.0176':10s} {res['mse_h1'].mean():.4f}")
print(f"{'H2 MSE':25s} {'0.0234':10s} {res['mse_h2'].mean():.4f}")
print(f"{'H2/H1 ratio':25s} {'1.33×':10s} {res['mse_h2'].mean()/res['mse_h1'].mean():.2f}×")
print(f"{'Mode sep':25s} {'0.838':10s} {res['sep']:.3f}")
print(f"{'r(z_state,Twc)':25s} {'0.679':10s} {res['r_state']:.3f}")
print(f"{'r(z_lag,dQH/dt)':25s} {'0.003':10s} {res['r_lag']:.3f}  ← KEY")
print(f"{'Ablation Δ MSE':25s} {'n/a':10s} +{(mse_ablated.mean()-res['mse_h2'].mean())/res['mse_h2'].mean()*100:.1f}%")
print(f"{'Case (delay)':25s} {'n/a':10s} {'B/C' if (mse_ablated.mean()-res['mse_h2'].mean())/res['mse_h2'].mean()>0.05 else 'A'}")
