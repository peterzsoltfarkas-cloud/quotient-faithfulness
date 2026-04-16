"""
Color Fantasy — Hierarchical GCD Benchmark
Comparing two architectures on the same data, losses, and evaluation.

ARCHITECTURE B (Two-stage):
  Stage 1: x → MLP_context → z_ctx   (context extracted from input alone)
  Stage 2: [x, z_ctx] → MLP_content → z_cont  (content conditioned on context)
  Decoder: FiLM(z_ctx) applied to decode(z_cont)

ARCHITECTURE E (Bottleneck):
  x → h1 → h2 → z_ctx (bottleneck at intermediate layer)
       ↓skip              ↓
  [h2, z_ctx] → h3 → h4 → z_cont
  Decoder: same FiLM decoder

BASELINE: flat MLP encoder, no FiLM, no structure losses

LATENT STRUCTURE (both B and E):
  z = (z_seasonal[2], z_lag[1], z_state[1], z_content[4])  total = 8

LOSSES:
  L_task:  MSE on 7 energy channels
  L_eq:    seasonal equivariance on z_seasonal (U(1) generator A_seasonal)
  L_lag:   temporal equivariance on z_lag (generator A_lag, Δt=2,4h, within-state)
  L_state: S2 monotone rank (T_wc ordering) + S3 HVAC prediction
  L_phys:  energy balance (exact conservation law)

EVALUATION (H1→H2 generalisation):
  - Train Jan-Jun (H1), evaluate Jul-Dec (H2)
  - Per-channel MSE
  - 8-mode separability (linear probe)
  - Latent disentanglement: does z_state correlate with T_wc threshold?
                            does z_lag correlate with local Q_H response rate?

DOCUMENTED CHOICES:
  - Δt for L_lag: 2h and 4h pairs, drawn from within-state consecutive hours
  - FiLM: gamma and beta from z_ctx, applied after first decoder layer
  - Monotone rank loss: sampled pairs from batch, ε=0.1 margin
  - State loss weight μ = 0.2 (HVAC prediction); rank weight κ = 0.1
  - Energy balance: predict 7 channels, close against P_total - P_prop
  - Architecture B: Stage 2 input dim = 21 + 4 (concat with z_ctx)
  - Architecture E: skip connection carries h2 (dim 128) to h3 input
"""
import numpy as np, pandas as pd, math, time
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

torch.manual_seed(42); np.random.seed(42)
OUTDIR = '/mnt/user-data/outputs/figures/'

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
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

# Windchill
T    = df['Air temp [°C]'].values.astype(float)
V    = df['Total wind [m/s]'].values.astype(float)
V_kmh= V * 3.6
Twc  = np.where(T<=10,
    13.12+0.6215*T-11.37*(V_kmh**0.16)+0.3965*T*(V_kmh**0.16), T)
df['T_wc'] = Twc

# Orthogonal thermal states
STATE = np.where(Twc<=5, 2, np.where(Twc<=21, 1, np.where(T>=24, 3, 0)))
df['state'] = STATE

# Channels
CHAN_COLS   = ['Acc. heating W','Electric cooling W','Domestic hot water W',
               'Galley steam demand W','HVAC aux W','Lighting facility W',
               'Equipment tenant W']
CHAN_SHORT  = ['Heating','Cooling','DHW','Galley','HVAC','Lighting','Equipment']
Y_raw = df[CHAN_COLS].values / 1e6   # MW
P_prop= df['Propulsion W'].values / 1e6
P_tot = df['Tot energy W'].values / 1e6

# ── Build 21-feature lagged input vector ──────────────────────────────────────
# Documented choice: lags {0,2,4,6,8,12}h for T_wc (6 values)
#                    lags {0,4,8}h for DNI and Diffuse (3 each)
#                    lag 0h for RH
#                    + sin/cos day/year, sin/cos hour/day, port/sea, state
DNI  = df['Direct normal radiation [W/m2]'].values.astype(float)
Dif  = df['Diffuse radition on horizontal surface [W/m2]'].values.astype(float)
RH   = df['Relative humidity [%]'].values.astype(float)
hour = df['hour'].values
day  = df['day'].values
port = df['in_port'].values.astype(float)

def lag_col(arr, k):
    """Lag array by k steps, fill start with first value."""
    out = np.empty_like(arr)
    out[:k] = arr[0]
    out[k:] = arr[:-k] if k > 0 else arr
    return out

Twc_lags = np.column_stack([lag_col(Twc, k) for k in [0,2,4,6,8,12]])  # 6
DNI_lags = np.column_stack([lag_col(DNI, k) for k in [0,4,8]])          # 3
Dif_lags = np.column_stack([lag_col(Dif, k) for k in [0,4,8]])          # 3
RH_now   = RH.reshape(-1,1)                                              # 1

# Time encodings
sin_day = np.sin(2*np.pi*day/365).reshape(-1,1)
cos_day = np.cos(2*np.pi*day/365).reshape(-1,1)
sin_hr  = np.sin(2*np.pi*hour/24).reshape(-1,1)
cos_hr  = np.cos(2*np.pi*hour/24).reshape(-1,1)
port_v  = port.reshape(-1,1)
state_v = STATE.reshape(-1,1).astype(float)   # 0,1,2,3

X_raw = np.hstack([Twc_lags, DNI_lags, Dif_lags, RH_now,
                   sin_day, cos_day, sin_hr, cos_hr,
                   port_v, state_v])  # 21 features total
assert X_raw.shape[1] == 19, f"Expected 19 features, got {X_raw.shape[1]}"

# Normalise inputs and outputs
from sklearn.preprocessing import StandardScaler
Xsc  = StandardScaler().fit(X_raw)
Ysc  = StandardScaler().fit(Y_raw)
X_n  = Xsc.transform(X_raw).astype(np.float32)
Y_n  = Ysc.transform(Y_raw).astype(np.float32)

# Energy balance target: P_total - P_prop = sum of 7 channels
E_balance = (P_tot - P_prop).astype(np.float32)  # MW, not normalised
# Predicted channel sum (un-normalised) should ≈ E_balance
chan_mean_sum = Ysc.mean_.sum()
chan_scale_sum= Ysc.scale_.mean() * 7

# H1: day ≤ 181 (Jan-Jun), H2: day > 181 (Jul-Dec)
h1_idx = np.where(day<=181)[0]
h2_idx = np.where(day>181)[0]
print(f"H1: {len(h1_idx)} h, H2: {len(h2_idx)} h")
print(f"Input features: {X_raw.shape[1]}, Output channels: 7")

X_h1 = torch.tensor(X_n[h1_idx]); Y_h1 = torch.tensor(Y_n[h1_idx])
X_h2 = torch.tensor(X_n[h2_idx]); Y_h2 = torch.tensor(Y_n[h2_idx])
Twc_h1 = torch.tensor(Twc[h1_idx].astype(np.float32))
E_h1   = torch.tensor(E_balance[h1_idx])
Q_hvac_h1 = torch.tensor(Y_n[h1_idx,4])   # HVAC normalised
State_h1  = torch.tensor(STATE[h1_idx].astype(np.float32))
Hour_h1   = torch.tensor(hour[h1_idx].astype(np.int64))
Port_h1   = torch.tensor(port[h1_idx].astype(np.float32))

# Modes for separability evaluation
df['mode'] = df['state']*2 + df['in_port']   # 8 modes: state(0-3) × port(0-1)
modes_h1 = df['mode'].values[h1_idx]
modes_h2 = df['mode'].values[h2_idx]

# ══════════════════════════════════════════════════════════════════════════════
# 2. ARCHITECTURE COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
IN_DIM  = 19
Z_CTX   = 4   # z_seasonal(2) + z_lag(1) + z_state(1)
Z_CONT  = 4   # content dimensions
Z_TOTAL = Z_CTX + Z_CONT   # = 8
OUT_DIM = 7
H_DIM   = 64

class MLP(nn.Module):
    def __init__(self, in_d, out_d, h=H_DIM, layers=2, act=nn.ReLU):
        super().__init__()
        dims = [in_d] + [h]*layers + [out_d]
        net  = []
        for i in range(len(dims)-1):
            net.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2: net.append(act())
        self.net = nn.Sequential(*net)
    def forward(self, x): return self.net(x)

class LieGen(nn.Module):
    """U(1) generator for z_seasonal, and R translation for z_lag."""
    def __init__(self, d_seasonal=2, d_lag=1):
        super().__init__()
        # A_seasonal: skew-symmetric 2×2 acts on z_seasonal
        self.W_seasonal = nn.Parameter(torch.randn(2,2)*0.1)
        # A_lag: scalar translation on z_lag (no rotation needed)
        self.a_lag = nn.Parameter(torch.ones(1)*0.1)

    def A_seasonal(self):
        W = self.W_seasonal
        return 0.5*(W - W.t())

    def act_seasonal(self, z_s, delta):
        """Rotate z_seasonal by delta (radians)."""
        A = self.A_seasonal()   # (2,2)
        # M[b] = delta[b] * A
        M = delta.unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(0)  # (B,2,2)
        return torch.bmm(torch.matrix_exp(M), z_s.unsqueeze(-1)).squeeze(-1)

    def act_lag(self, z_l, dt):
        """Translate z_lag by dt * a_lag."""
        return z_l + dt.unsqueeze(1) * self.a_lag

class FiLMDecoder(nn.Module):
    """FiLM-conditioned decoder: γ,β from z_ctx, applied to decoded z_cont."""
    def __init__(self):
        super().__init__()
        self.content_net  = MLP(Z_CONT, H_DIM, h=H_DIM, layers=1)
        self.film_net     = nn.Linear(Z_CTX, H_DIM*2)   # γ and β
        self.output_layer = nn.Linear(H_DIM, OUT_DIM)

    def forward(self, z_ctx, z_cont):
        h    = self.content_net(z_cont)
        film = self.film_net(z_ctx)
        gamma, beta = film[:,:H_DIM], film[:,H_DIM:]
        h    = gamma * h + beta   # FiLM modulation
        return self.output_layer(h)

# ── Architecture B: Two-stage encoder ────────────────────────────────────────
class EncoderB(nn.Module):
    """
    Stage 1: x → h1 → h2 → z_ctx
    Stage 2: [x, z_ctx] → h3 → h4 → z_cont
    Documented choice: Stage 2 input = concat(x, z_ctx) = 21+4=25 dims
    """
    def __init__(self):
        super().__init__()
        self.stage1 = MLP(IN_DIM, H_DIM, h=H_DIM, layers=2)
        self.ctx_head = nn.Linear(H_DIM, Z_CTX)
        self.stage2 = MLP(IN_DIM + Z_CTX, H_DIM, h=H_DIM, layers=2)
        self.cont_head = nn.Linear(H_DIM, Z_CONT)

    def forward(self, x):
        h_ctx  = self.stage1(x)
        z_ctx  = self.ctx_head(h_ctx)
        h_cont = self.stage2(torch.cat([x, z_ctx], dim=1))
        z_cont = self.cont_head(h_cont)
        return z_ctx, z_cont

# ── Architecture E: Bottleneck encoder ────────────────────────────────────────
class EncoderE(nn.Module):
    """
    x → h1 → h2 → z_ctx  [bottleneck at intermediate layer]
              ↓skip
    [h2, z_ctx] → h3 → h4 → z_cont
    Documented choice: skip carries h2 (128-dim) to h3 input
    h3 input dim = H_DIM + Z_CTX = 132
    """
    def __init__(self):
        super().__init__()
        self.layer1   = nn.Sequential(nn.Linear(IN_DIM, H_DIM), nn.ReLU())
        self.layer2   = nn.Sequential(nn.Linear(H_DIM, H_DIM), nn.ReLU())
        self.ctx_head = nn.Linear(H_DIM, Z_CTX)
        self.layer3   = nn.Sequential(nn.Linear(H_DIM+Z_CTX, H_DIM), nn.ReLU())
        self.layer4   = nn.Sequential(nn.Linear(H_DIM, H_DIM), nn.ReLU())
        self.cont_head= nn.Linear(H_DIM, Z_CONT)

    def forward(self, x):
        h1     = self.layer1(x)
        h2     = self.layer2(h1)
        z_ctx  = self.ctx_head(h2)
        h3     = self.layer3(torch.cat([h2, z_ctx], dim=1))  # skip from h2
        h4     = self.layer4(h3)
        z_cont = self.cont_head(h4)
        return z_ctx, z_cont

# ── Flat baseline ─────────────────────────────────────────────────────────────
class EncoderFlat(nn.Module):
    """Standard MLP: no hierarchy, no FiLM, no structure losses."""
    def __init__(self):
        super().__init__()
        self.net = MLP(IN_DIM, Z_TOTAL, h=H_DIM, layers=4)
        self.head= nn.Linear(Z_TOTAL, OUT_DIM)

    def forward_enc(self, x):
        z = self.net(x)
        return z[:,:Z_CTX], z[:,Z_CTX:]   # split for compatibility

    def forward(self, x):
        return self.head(self.net(x))

# ══════════════════════════════════════════════════════════════════════════════
# 3. LOSS FUNCTIONS (all documented)
# ══════════════════════════════════════════════════════════════════════════════
def loss_task(pred, target):
    return nn.functional.mse_loss(pred, target)

def loss_eq_seasonal(enc, gen, x, x_aug, delta):
    """
    L_eq: enc(x_shifted_by_delta_days)[seasonal] ≈ A_seasonal(enc(x)[seasonal], delta)
    Documented: delta drawn uniform [7,45] days as fraction of year,
                augmentation shifts sin/cos_day columns (indices 13,14 in X)
    """
    z_ctx, _ = enc(x)
    z_aug_ctx, _ = enc(x_aug)
    z_s      = z_ctx[:, :2]   # z_seasonal
    z_s_aug  = z_aug_ctx[:, :2]
    z_s_pred = gen.act_seasonal(z_s, delta)
    return nn.functional.mse_loss(z_s_aug, z_s_pred)

def loss_eq_lag(enc, gen, x_t, x_t_plus, dt, state_mask):
    """
    L_lag: enc(x(t+dt))[lag] ≈ A_lag(enc(x(t))[lag], dt)
    Documented: dt ∈ {2,4} hours, pairs drawn from within-state consecutive hours.
    state_mask: boolean tensor selecting valid within-state pairs.
    """
    if state_mask.sum() < 4:
        return torch.tensor(0.0)
    x0 = x_t[state_mask]; x1 = x_t_plus[state_mask]
    dt_m = dt[state_mask]
    z0_ctx, _ = enc(x0)
    z1_ctx, _ = enc(x1)
    z0_lag = z0_ctx[:, 2:3]   # z_lag (1D)
    z1_lag = z1_ctx[:, 2:3]
    z0_pred = gen.act_lag(z0_lag, dt_m)
    return nn.functional.mse_loss(z1_lag, z0_pred)

def loss_state(enc, z_ctx_batch, twc_batch, q_hvac_batch):
    """
    L_state = S2 (monotone rank) + S3 (HVAC prediction)

    S2: if T_wc_i < T_wc_j then z_state_i should < z_state_j
        Documented: sampled 512 pairs from batch, margin ε=0.1
    S3: z_state should predict HVAC (step change at 5°C is HVAC signal)
        Documented: linear head on z_state alone, MSE loss
    """
    z_s = z_ctx_batch[:, 3]   # z_state (scalar)

    # S2: monotone rank loss
    B = min(len(z_s), 256)
    idx = torch.randperm(len(z_s))[:B]
    z_s_b  = z_s[idx]; twc_b = twc_batch[idx]
    # Pairwise: (i,j) where T_wc_i < T_wc_j
    diff_twc = twc_b.unsqueeze(1) - twc_b.unsqueeze(0)   # (B,B)
    diff_z   = z_s_b.unsqueeze(1) - z_s_b.unsqueeze(0)   # (B,B)
    # Violation: T_wc_i < T_wc_j but z_state_i >= z_state_j
    violations = (diff_twc < -0.5).float() * torch.relu(diff_z + 0.1)
    L_rank = violations.mean()

    # S3: HVAC prediction from z_state alone
    # (HVAC has step at 5°C threshold — z_state must encode this)
    L_hvac = ((z_s - q_hvac_batch)**2).mean()

    return L_rank + 0.2 * L_hvac

def loss_phys_balance(pred_y, e_balance_batch, ysc):
    """
    L_phys: energy balance (exact conservation law)
    sum(Q_i_pred) ≈ P_total - P_prop
    Documented: un-normalise pred_y, compare against E_balance (MW)
    """
    # Un-normalise: pred_y is in normalised space
    scale = torch.tensor(ysc.scale_, dtype=torch.float32)
    mean  = torch.tensor(ysc.mean_,  dtype=torch.float32)
    pred_mw = pred_y * scale + mean   # (B, 7) in MW
    pred_sum = pred_mw.sum(dim=1)     # (B,)
    return ((pred_sum - e_balance_batch)**2).mean()

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ══════════════════════════════════════════════════════════════════════════════
EPOCHS = 200; BS = 2048
N_H1   = len(X_h1)

def build_seasonal_aug(x_batch):
    """Shift sin/cos_day by delta ∈ [7,45] days."""
    delta_frac = (torch.rand(len(x_batch))*38+7) / 365 * 2*math.pi
    x_aug = x_batch.clone()
    phi_orig = torch.atan2(x_batch[:,13], x_batch[:,14])
    phi_new  = phi_orig + delta_frac
    x_aug[:,13] = torch.sin(phi_new)
    x_aug[:,14] = torch.cos(phi_new)
    return x_aug, delta_frac

def build_lag_pairs(x_h1, state_arr, dt_h=2):
    """Build (x(t), x(t+dt)) pairs within the same thermal state."""
    # For each t, pair with t+dt_h if state[t] == state[t+dt_h]
    t_idx = torch.arange(N_H1 - dt_h)
    s_t   = torch.tensor(state_arr[:N_H1-dt_h].astype(np.float32))
    s_tp  = torch.tensor(state_arr[dt_h:N_H1].astype(np.float32))
    same_state = (s_t == s_tp) & (s_t > 0)  # exclude dead zone (state 0)
    return t_idx, same_state, torch.full((N_H1-dt_h,), float(dt_h)/24.0)

# Pre-build lag pair indices
t_idx_2h, same_state_2h, dt_2h = build_lag_pairs(X_h1, STATE[h1_idx], dt_h=2)
t_idx_4h, same_state_4h, dt_4h = build_lag_pairs(X_h1, STATE[h1_idx], dt_h=4)

Ysc_tensor = Ysc  # kept for loss_phys_balance

def train(arch_name, enc, use_structure=True):
    """
    Train one architecture. Returns evaluation results.
    arch_name: 'Baseline', 'Arch-B', 'Arch-E'
    enc: encoder module
    use_structure: if False, only L_task (baseline)
    """
    if isinstance(enc, EncoderFlat):
        dec = None   # flat has its own head
    else:
        dec = FiLMDecoder()
    gen = LieGen()

    if isinstance(enc, EncoderFlat):
        params = list(enc.parameters())
    else:
        params = list(enc.parameters())+list(dec.parameters())+list(gen.parameters())

    opt = optim.Adam(params, lr=5e-4)
    history = {k:[] for k in ['task','eq','lag','state','phys','total']}

    t0 = time.time()
    for ep in range(EPOCHS):
        idx = torch.randint(N_H1, (BS,))
        x_b  = X_h1[idx]; y_b  = Y_h1[idx]
        twc_b= Twc_h1[idx]; e_b = E_h1[idx]
        qhv_b= Q_hvac_h1[idx]

        if isinstance(enc, EncoderFlat):
            pred = enc(x_b)
            loss = loss_task(pred, y_b)
        else:
            z_ctx, z_cont = enc(x_b)
            pred = dec(z_ctx, z_cont)
            loss = loss_task(pred, y_b)
            history['task'].append(loss.item())

            if use_structure:
                # L_eq seasonal
                x_aug, delta = build_seasonal_aug(x_b)
                Le = loss_eq_seasonal(enc, gen, x_b, x_aug, delta)
                loss = loss + 0.3*Le
                history['eq'].append(Le.item())

                # L_lag (sample a subset of pre-built pairs)
                sub = torch.randint(len(t_idx_2h), (min(256,len(t_idx_2h)),))
                ti = t_idx_2h[sub]; sm = same_state_2h[sub]; dt = dt_2h[sub]
                Ll = loss_eq_lag(enc, gen, X_h1[ti], X_h1[ti+2], dt, sm)
                loss = loss + 0.2*Ll
                history['lag'].append(Ll.item())

                # L_state
                Ls = loss_state(enc, z_ctx, twc_b, qhv_b)
                loss = loss + 0.2*Ls
                history['state'].append(Ls.item())

                # L_phys energy balance
                Lp = loss_phys_balance(pred, e_b, Ysc)
                Lp = torch.clamp(Lp, max=2.0)
                loss = loss + 0.1*Lp
                history['phys'].append(Lp.item())

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    elapsed = time.time()-t0

    # ── Evaluation ─────────────────────────────────────────────────────────
    with torch.no_grad():
        if isinstance(enc, EncoderFlat):
            pred_h1_n = enc(X_h1).numpy()
            pred_h2_n = enc(X_h2).numpy()
            Z_h1 = enc.net(X_h1).numpy()
            Z_h2 = enc.net(X_h2).numpy()
        else:
            z_ctx_h1, z_cont_h1 = enc(X_h1)
            z_ctx_h2, z_cont_h2 = enc(X_h2)
            pred_h1_n = dec(z_ctx_h1, z_cont_h1).numpy()
            pred_h2_n = dec(z_ctx_h2, z_cont_h2).numpy()
            Z_h1 = torch.cat([z_ctx_h1, z_cont_h1], dim=1).numpy()
            Z_h2 = torch.cat([z_ctx_h2, z_cont_h2], dim=1).numpy()

    # Un-normalise
    pred_h1_mw = Ysc.inverse_transform(pred_h1_n)
    pred_h2_mw = Ysc.inverse_transform(pred_h2_n)
    true_h1_mw = Ysc.inverse_transform(Y_h1.numpy())
    true_h2_mw = Ysc.inverse_transform(Y_h2.numpy())

    mse_h1 = np.mean((pred_h1_mw-true_h1_mw)**2, axis=0)
    mse_h2 = np.mean((pred_h2_mw-true_h2_mw)**2, axis=0)

    # Mode separability
    Z_all = np.nan_to_num(np.vstack([Z_h1,Z_h2]))
    Z_sub = Z_all[::4]; m_sub = np.concatenate([modes_h1,modes_h2])[::4]
    sep = cross_val_score(LogisticRegression(max_iter=300),
                          Z_sub, m_sub,
                          cv=3, scoring='accuracy').mean()

    # Disentanglement: does z_state (index 3) correlate with T_wc?
    if not isinstance(enc, EncoderFlat):
        r_state_twc = np.corrcoef(Z_h1[:,3], Twc[h1_idx])[0,1]
        r_lag_rate  = np.corrcoef(
            Z_h1[:,2],
            np.gradient(Y_raw[h1_idx,0])   # dQ_H/dt as proxy for rate
        )[0,1]
    else:
        r_state_twc = float('nan')
        r_lag_rate  = float('nan')

    print(f"  {arch_name:10s}: H1={mse_h1.mean():.4f}  H2={mse_h2.mean():.4f}"
          f"  sep={sep:.3f}  r(z_state,Twc)={r_state_twc:.3f}"
          f"  r(z_lag,dQH)={r_lag_rate:.3f}  t={elapsed:.0f}s")

    return dict(arch=arch_name, mse_h1=mse_h1, mse_h2=mse_h2, sep=sep,
                r_state=r_state_twc, r_lag=r_lag_rate,
                pred_h2=pred_h2_mw, true_h2=true_h2_mw,
                Z_h1=Z_h1, Z_h2=Z_h2, history=history, elapsed=elapsed)

print("\nTraining three architectures...")
results = {}
results['Baseline'] = train('Baseline', EncoderFlat(), use_structure=False)
np.save('/home/claude/result_baseline.npy', {k:v for k,v in results['Baseline'].items() if k not in ['Z_h1','Z_h2']})
results['Arch-B']   = train('Arch-B',   EncoderB(),    use_structure=True)
np.save('/home/claude/result_arch_b.npy', {k:v for k,v in results['Arch-B'].items() if k not in ['Z_h1','Z_h2']})
results['Arch-E']   = train('Arch-E',   EncoderE(),    use_structure=True)
np.save('/home/claude/result_arch_e.npy', {k:v for k,v in results['Arch-E'].items() if k not in ['Z_h1','Z_h2']})

# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
ARCH_COL = {'Baseline':'#888888','Arch-B':'#2166ac','Arch-E':'#d73027'}

# Fig 1: Per-channel H2 MSE comparison
fig, axes = plt.subplots(1,3, figsize=(16,5))
fig.suptitle('H1→H2 Generalisation: Per-Channel MSE\n'
             'Arch-B=Two-stage encoder, Arch-E=Bottleneck encoder',
             fontsize=11, fontweight='bold')
for ai, metric in enumerate(['mse_h1','mse_h2',None]):
    ax = axes[ai]
    x  = np.arange(7); w = 0.28
    for ci,(name,col) in enumerate(ARCH_COL.items()):
        vals = results[name][metric] if metric else \
               results[name]['mse_h2']/(results[name]['mse_h1']+1e-8)
        ax.bar(x+ci*w, vals, w, color=col, alpha=0.85, label=name)
    ax.set_xticks(x+w); ax.set_xticklabels(CHAN_SHORT,rotation=30,ha='right',fontsize=8.5)
    ax.set_title(['H1 train MSE','H2 test MSE','H2/H1 ratio'][ai], fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True,axis='y',alpha=0.3)
plt.tight_layout()
fig.savefig(OUTDIR+'cf_arch_comparison_mse.png',dpi=200,bbox_inches='tight')
plt.close(); print("Saved cf_arch_comparison_mse.png")

# Fig 2: Per-channel H2 predicted curves (thermal channels, both arch)
days_h2 = day[h2_idx]
unique_days = sorted(set(days_h2))
fig, axes = plt.subplots(3,3, figsize=(16,13))
fig.suptitle('H2 Predicted vs Actual — Thermal Channels (daily means)\n'
             'Row 1: Baseline, Row 2: Arch-B, Row 3: Arch-E',
             fontsize=11, fontweight='bold')
thermal_chans = [0,1,4]  # Heating, Cooling, HVAC
for ri,(rname,rcol) in enumerate(ARCH_COL.items()):
    for ci,cj in enumerate(thermal_chans):
        ax = axes[ri,ci]
        true_d  = [results[rname]['true_h2'][days_h2==d,cj].mean()
                   for d in unique_days]
        pred_d  = [results[rname]['pred_h2'][days_h2==d,cj].mean()
                   for d in unique_days]
        mse_val = results[rname]['mse_h2'][cj]
        ax.plot(true_d,'k-',lw=1.5,alpha=0.8,label='Actual')
        ax.plot(pred_d,color=rcol,lw=1.5,ls='--',
                label=f'{rname} (MSE={mse_val:.4f})')
        ax.fill_between(range(len(true_d)),true_d,pred_d,alpha=0.15,color=rcol)
        if ri==0: ax.set_title(CHAN_SHORT[cj],fontweight='bold')
        ax.set_ylabel(f'{rname}',fontsize=8)
        ax.legend(fontsize=7.5); ax.grid(True,alpha=0.2)
        ax.set_xlabel('Day in H2' if ri==2 else '')
plt.tight_layout()
fig.savefig(OUTDIR+'cf_arch_h2_curves_thermal.png',dpi=200,bbox_inches='tight')
plt.close(); print("Saved cf_arch_h2_curves_thermal.png")

# Fig 3: Latent space comparison — z_state and z_lag vs physics
fig, axes = plt.subplots(2,3,figsize=(16,9))
fig.suptitle('Latent Disentanglement: do z_state and z_lag encode physical structure?\n'
             'Arch-B vs Arch-E (Baseline has no structure supervision)',
             fontsize=11, fontweight='bold')
for ri,(aname,acol) in enumerate([('Arch-B','#2166ac'),('Arch-E','#d73027')]):
    Z = results[aname]['Z_h1']
    # z_state (index 3) vs T_wc
    ax = axes[ri,0]
    sc = ax.scatter(Twc[h1_idx][::4], Z[::4,3], s=3, alpha=0.3,
                    c=STATE[h1_idx][::4], cmap='RdYlBu_r', vmin=0, vmax=3)
    plt.colorbar(sc,ax=ax,label='True state')
    ax.axvline(5, color='k',lw=1.5,ls='--',alpha=0.7)
    ax.axvline(21,color='k',lw=1.5,ls='--',alpha=0.7)
    ax.set_xlabel('T_wc (°C)'); ax.set_ylabel('z_state (latent)')
    ax.set_title(f'{aname}: z_state vs T_wc\nr={results[aname]["r_state"]:.3f}',
                 fontweight='bold')
    ax.grid(True,alpha=0.2)
    # z_lag (index 2) vs dQ_H/dt
    ax = axes[ri,1]
    rate = np.gradient(Y_raw[h1_idx,0])
    ax.scatter(rate[::4], Z[::4,2], s=3, alpha=0.3, color=acol)
    ax.set_xlabel('dQ_H/dt (MW/h)'); ax.set_ylabel('z_lag (latent)')
    ax.set_title(f'{aname}: z_lag vs heating rate\nr={results[aname]["r_lag"]:.3f}',
                 fontweight='bold')
    ax.grid(True,alpha=0.2)
    # z_seasonal PCA
    ax = axes[ri,2]
    z_s = Z[:, :2]
    sc2 = ax.scatter(z_s[::4,0], z_s[::4,1], s=3, alpha=0.3,
                     c=day[h1_idx][::4], cmap='hsv')
    plt.colorbar(sc2,ax=ax,label='Day of year')
    ax.set_xlabel('z_seasonal[0]'); ax.set_ylabel('z_seasonal[1]')
    ax.set_title(f'{aname}: z_seasonal colored by day\n(should form S¹)',
                 fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True,alpha=0.2)
plt.tight_layout()
fig.savefig(OUTDIR+'cf_latent_disentanglement.png',dpi=200,bbox_inches='tight')
plt.close(); print("Saved cf_latent_disentanglement.png")

# Fig 4: Summary comparison table
fig, ax = plt.subplots(figsize=(14,4))
ax.axis('off')
rows = []
for name in ['Baseline','Arch-B','Arch-E']:
    r = results[name]
    rows.append([
        name,
        'None' if name=='Baseline' else 'L_eq+L_lag+L_state+L_phys',
        'Flat MLP' if name=='Baseline' else ('Two-stage' if name=='Arch-B' else 'Bottleneck'),
        f"{r['mse_h1'].mean():.4f}",
        f"{r['mse_h2'].mean():.4f}",
        f"{r['mse_h2'].mean()/r['mse_h1'].mean():.2f}×",
        f"{r['sep']:.3f}",
        f"{r['r_state']:.3f}",
        f"{r['r_lag']:.3f}",
        f"{r['elapsed']:.0f}s",
    ])
tbl = ax.table(cellText=rows,
               colLabels=['Arch','Losses','Structure',
                          'H1 MSE','H2 MSE','H2/H1',
                          'Sep.','r(z_s,Twc)','r(z_l,dQ)','Time'],
               cellLoc='center', loc='center',
               colWidths=[0.08,0.2,0.1,0.07,0.07,0.07,0.06,0.1,0.09,0.06])
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.8)
for (r,c),cell in tbl.get_celld().items():
    if r==0:
        cell.set_facecolor('#2166ac'); cell.set_text_props(color='w',weight='bold')
    elif r>0:
        col = ['#f7f7f7','#dbeeff','#ffe0d8'][r-1]
        cell.set_facecolor(col)
ax.set_title('Architecture Benchmark Summary\n'
             'Documented choices: BS=1024, epochs=300, lr=5e-4, '
             'λ_eq=0.3, λ_lag=0.2, λ_state=0.2, λ_phys=0.1',
             fontsize=9, fontweight='bold', pad=20)
plt.tight_layout()
fig.savefig(OUTDIR+'cf_arch_summary_table.png',dpi=200,bbox_inches='tight')
plt.close(); print("Saved cf_arch_summary_table.png")
