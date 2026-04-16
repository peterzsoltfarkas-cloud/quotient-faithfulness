"""
Persistent homology certification for all three benchmark experiments.
Trains GCD and baseline encoders, extracts latent point clouds,
runs ripser, and reports circ(Z) and torus(Z) scores.
"""
import torch, math, numpy as np
import torch.nn as nn, torch.optim as optim
from ripser import ripser as rips
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42); np.random.seed(42)
THRESH_S1  = 0.5   # circ(Z) > 0.5 → S^1 topology certified
THRESH_T2  = 0.3   # two bars > 0.3 → T^2 topology certified
N_PH_PTS   = 500   # points for PH (manageable for ripser)

# ── Architecture ──────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, layers=3):
        super().__init__()
        net = []; dims = [in_dim]+[hidden]*(layers-1)+[out_dim]
        for i in range(len(dims)-1):
            net.append(nn.Linear(dims[i],dims[i+1]))
            if i < len(dims)-2: net.append(nn.ReLU())
        self.net = nn.Sequential(*net)
    def forward(self,x): return self.net(x)

class LieGen(nn.Module):
    def __init__(self,d,k=1):
        super().__init__()
        self.d,self.k=d,k
        self.W=nn.Parameter(torch.randn(k,d,d)*0.1)
    def generators(self):
        W=self.W; return 0.5*(W-W.transpose(-1,-2))
    def action(self,z,c):
        if c.dim()==1: c=c.unsqueeze(0).expand(z.size(0),-1)
        M=torch.einsum('bk,kij->bij',c,self.generators())
        return torch.einsum('bij,bj->bi',torch.matrix_exp(M),z)
    def closure_loss(self):
        A=self.generators()
        comms=[A[i]@A[j]-A[j]@A[i]
               for i in range(self.k) for j in range(i+1,self.k)]
        if not comms: return torch.tensor(0.)
        return torch.stack(comms).pow(2).mean()

def eq_loss(enc,gen,x,xa,c):
    return nn.functional.mse_loss(enc(xa),gen.action(enc(x),c))

def circ(Z):
    """Longest H1 bar lifetime."""
    Zn = StandardScaler().fit_transform(Z)
    dgms = rips(Zn, maxdim=1)['dgms']
    h1   = dgms[1]
    if len(h1) == 0: return 0.0
    h1_fin = h1[~np.isinf(h1[:,1])]
    if len(h1_fin) == 0: return 0.0
    return float(np.max(h1_fin[:,1] - h1_fin[:,0]))

def torus(Z, tau=THRESH_T2):
    """Number of persistent H1 bars above threshold."""
    Zn = StandardScaler().fit_transform(Z)
    dgms = rips(Zn, maxdim=1)['dgms']
    h1   = dgms[1]
    h1_fin = h1[~np.isinf(h1[:,1])]
    count = int(np.sum(h1_fin[:,1]-h1_fin[:,0] > tau))
    return count

def top_bars(Z, n=3):
    """Return top n H1 bar lifetimes."""
    Zn = StandardScaler().fit_transform(Z)
    dgms = rips(Zn, maxdim=1)['dgms']
    h1   = dgms[1]
    h1_fin = h1[~np.isinf(h1[:,1])]
    lifetimes = sorted(h1_fin[:,1]-h1_fin[:,0], reverse=True)
    return lifetimes[:n]

# ═════════════════════════════════════════════════════════════════════════════
# EXP 1: PHASE CYLINDER
# ═════════════════════════════════════════════════════════════════════════════
print("="*60)
print("EXP 1: Phase Cylinder  (Q = S^1)")
print("="*60)
torch.manual_seed(42); np.random.seed(42)

def pc_batch(B=256):
    theta=torch.rand(B)*2*math.pi
    x=torch.stack([torch.cos(theta),torch.sin(theta)],1)+0.05*torch.randn(B,2)
    delta=(torch.rand(B)-0.5)*0.5
    xa=torch.stack([torch.cos(theta+delta),torch.sin(theta+delta)],1)+0.05*torch.randn(B,2)
    return x,xa,delta.unsqueeze(1),theta.unsqueeze(1)

# Baseline m=1
eb=MLP(2,1,hidden=64); hb=nn.Linear(1,1)
ob=optim.Adam(list(eb.parameters())+list(hb.parameters()),lr=1e-3)
for ep in range(500):
    x,xa,d,y=pc_batch()
    lt=((hb(eb(x))-y)**2).mean()
    ob.zero_grad();lt.backward();ob.step()

# GCD m=2
eg=MLP(2,2,hidden=64); gg=LieGen(2,1); hg=nn.Linear(2,1)
og=optim.Adam(list(eg.parameters())+list(gg.parameters())+list(hg.parameters()),lr=5e-4)
for ep in range(500):
    x,xa,d,y=pc_batch()
    z=eg(x); lt=((hg(z)-y)**2).mean()
    le=eq_loss(eg,gg,x,xa,d); lc=gg.closure_loss()
    loss=lt+0.5*le+0.05*lc
    og.zero_grad();loss.backward();og.step()

# Extract latent point clouds
with torch.no_grad():
    test_pts = []
    for _ in range(20):
        x,_,_,_ = pc_batch(256)
        test_pts.append(x)
    Xtest = torch.cat(test_pts)[:N_PH_PTS]
    Z_base = eb(Xtest).numpy()
    Z_gcd  = eg(Xtest).numpy()

circ_base = circ(Z_base)
circ_gcd  = circ(Z_gcd)
bars_base = top_bars(Z_base)
bars_gcd  = top_bars(Z_gcd)

print(f"  Baseline (m=1): circ(Z) = {circ_base:.4f}  top bars: {[f'{b:.3f}' for b in bars_base]}")
print(f"  GCD     (m=2): circ(Z) = {circ_gcd:.4f}  top bars: {[f'{b:.3f}' for b in bars_gcd]}")
print(f"  S^1 certified (circ > {THRESH_S1}):")
print(f"    Baseline: {'YES' if circ_base > THRESH_S1 else 'NO'}")
print(f"    GCD:      {'YES' if circ_gcd  > THRESH_S1 else 'NO'}")

# ═════════════════════════════════════════════════════════════════════════════
# EXP 2: HOPF FIBRATION
# ═════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("EXP 2: Hopf Fibration  (Q = S^2)")
print("="*60)
torch.manual_seed(42); np.random.seed(42)

def hopf(x):
    z1r,z1i,z2r,z2i=x[:,0],x[:,1],x[:,2],x[:,3]
    return torch.stack([2*(z1r*z2r+z1i*z2i),
                        2*(z1i*z2r-z1r*z2i),
                        z1r**2+z1i**2-z2r**2-z2i**2],1)

def hf_batch(B=256):
    x=torch.randn(B,4);x=x/x.norm(dim=1,keepdim=True)
    t=torch.rand(B)*0.3;c,s=torch.cos(t),torch.sin(t)
    xa=torch.stack([c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1],
                    c*x[:,2]-s*x[:,3],s*x[:,2]+c*x[:,3]],1)
    return x,xa,t.unsqueeze(1),hopf(x)

# Baseline m=2
ebh=MLP(4,2,hidden=128); hbh=nn.Linear(2,3)
obh=optim.Adam(list(ebh.parameters())+list(hbh.parameters()),lr=1e-3)
for ep in range(500):
    x,xa,t,y=hf_batch()
    lt=((hbh(ebh(x))-y)**2).mean()
    obh.zero_grad();lt.backward();obh.step()

# GCD m=3
egh=MLP(4,3,hidden=128); ggh=LieGen(3,1); hgh=nn.Linear(3,3)
ogh=optim.Adam(list(egh.parameters())+list(ggh.parameters())+list(hgh.parameters()),lr=1e-3)
for ep in range(500):
    x,xa,t,y=hf_batch()
    z=egh(x); lt=((hgh(z)-y)**2).mean()
    le=eq_loss(egh,ggh,x,xa,t); lc=ggh.closure_loss()
    loss=lt+2.0*le+0.2*lc
    ogh.zero_grad();loss.backward();ogh.step()

with torch.no_grad():
    test_pts=[]
    for _ in range(20):
        x,_,_,_=hf_batch(256)
        test_pts.append(x)
    Xtest=torch.cat(test_pts)[:N_PH_PTS]
    Zh_base=ebh(Xtest).numpy()
    Zh_gcd =egh(Xtest).numpy()

# For Hopf, GCD should produce S^2 topology. 
# S^2 has H2 class (not H1), but we can check H1=0 and H2>0.
# Use ripser with maxdim=2 for the GCD latent.
def sphericity(Z):
    """Check for S^2: H1=0 (no persistent loop), H2 > 0 (persistent sphere)."""
    Zn = StandardScaler().fit_transform(Z)
    dgms = rips(Zn, maxdim=2)['dgms']
    h1 = dgms[1]; h2 = dgms[2]
    h1_fin = h1[~np.isinf(h1[:,1])]
    h2_fin = h2[~np.isinf(h2[:,1])]
    max_h1 = float(np.max(h1_fin[:,1]-h1_fin[:,0])) if len(h1_fin) else 0.0
    max_h2 = float(np.max(h2_fin[:,1]-h2_fin[:,0])) if len(h2_fin) else 0.0
    return max_h1, max_h2

circ_hbase = circ(Zh_base)
h1_hgcd, h2_hgcd = sphericity(Zh_gcd)
print(f"  Baseline (m=2): circ(Z) = {circ_hbase:.4f}  (expected ≈ 0, dimension trap)")
print(f"  GCD     (m=3): max H1 = {h1_hgcd:.4f}  max H2 = {h2_hgcd:.4f}")
print(f"  S^2 certified (H2 persistent, H1 ≈ 0):")
print(f"    GCD: H1={h1_hgcd:.3f}, H2={h2_hgcd:.3f} → "
      f"{'CERTIFIED' if h2_hgcd > 0.1 and h1_hgcd < 0.3 else 'NOT certified'}")

# ═════════════════════════════════════════════════════════════════════════════
# EXP 3: DOUBLE PENDULUM
# ═════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("EXP 3: Double Pendulum  (Q = T^2)")
print("="*60)
torch.manual_seed(42); np.random.seed(42)

def dp_deriv(s,g=9.8):
    th1,th2,w1,w2=s;d=th2-th1;dm=2-np.cos(2*d)
    dw1=(-g*(2*np.sin(th1)+np.sin(th1-2*th2))-2*np.sin(d)*(w2**2+w1**2*np.cos(d)))/dm
    dw2=(2*np.sin(d)*(2*w1**2+2*g*np.cos(th1)+w2**2*np.cos(d)))/dm
    return np.array([w1,w2,dw1,dw2])

def rk4(s,dt=0.05):
    k1=dp_deriv(s);k2=dp_deriv(s+0.5*dt*k1)
    k3=dp_deriv(s+0.5*dt*k2);k4=dp_deriv(s+dt*k3)
    return s+(dt/6)*(k1+2*k2+2*k3+k4)

data=[]
for _ in range(60):
    s=np.array([np.random.uniform(-math.pi,math.pi),
                np.random.uniform(-math.pi,math.pi),
                np.random.uniform(-1,1),np.random.uniform(-1,1)])
    for _ in range(120): s=rk4(s); data.append(s.copy())
data=np.array(data,dtype=np.float32)

def enc_dp(d):
    th1,th2,w1,w2=d[:,0],d[:,1],d[:,2],d[:,3]
    return np.stack([np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),w1,w2],-1)
Xdp=torch.tensor(enc_dp(data))

def dp_batch(B=256):
    idx=torch.randint(len(Xdp),(B,));x=Xdp[idx]
    d1=(torch.rand(B)-0.5)*0.4;d2=(torch.rand(B)-0.5)*0.4
    c1,s1=torch.cos(d1),torch.sin(d1);c2,s2=torch.cos(d2),torch.sin(d2)
    x0=c1*x[:,0]-s1*x[:,1];x1=s1*x[:,0]+c1*x[:,1]
    x2=c2*x[:,2]-s2*x[:,3];x3=s2*x[:,2]+c2*x[:,3]
    return x,torch.stack([x0,x1,x2,x3,x[:,4],x[:,5]],1),torch.stack([d1,d2],1)

# Baseline m=2
ebd=MLP(6,2,hidden=128);hbd=nn.Linear(2,2)
obd=optim.Adam(list(ebd.parameters())+list(hbd.parameters()),lr=1e-3)
# GCD m=3
egd=MLP(6,3,hidden=128);ggd=LieGen(3,2);hgd=nn.Linear(3,2)
ogd=optim.Adam(list(egd.parameters())+list(ggd.parameters())+list(hgd.parameters()),lr=1e-3)

for ep in range(500):
    x,xa,d=dp_batch();y=x[:,4:6]
    lt=((hbd(ebd(x))-y)**2).mean()
    obd.zero_grad();lt.backward();obd.step()
    z=egd(x);lt=((hgd(z)-y)**2).mean()
    le=eq_loss(egd,ggd,x,xa,d);lc=ggd.closure_loss()
    loss=lt+1.5*le+0.1*lc
    ogd.zero_grad();loss.backward();ogd.step()

with torch.no_grad():
    Zd_base=ebd(Xdp[:N_PH_PTS]).numpy()
    Zd_gcd =egd(Xdp[:N_PH_PTS]).numpy()

circ_dbase = circ(Zd_base)
torus_dbase = torus(Zd_base)
bars_dbase  = top_bars(Zd_base, n=4)

circ_dgcd  = circ(Zd_gcd)
torus_dgcd = torus(Zd_gcd)
bars_dgcd  = top_bars(Zd_gcd, n=4)

print(f"  Baseline (m=2): circ={circ_dbase:.4f}  β1={torus_dbase}  top bars: {[f'{b:.3f}' for b in bars_dbase]}")
print(f"  GCD     (m=3): circ={circ_dgcd:.4f}   β1={torus_dgcd}  top bars: {[f'{b:.3f}' for b in bars_dgcd]}")
print(f"  T^2 certified (β1 ≥ 2, bars > {THRESH_T2}):")
print(f"    Baseline: β1={torus_dbase} → {'CERTIFIED' if torus_dbase >= 2 else 'NOT certified'}")
print(f"    GCD:      β1={torus_dgcd}  → {'CERTIFIED' if torus_dgcd  >= 2 else 'NOT certified'}")

print("\n" + "="*60)
print("SUMMARY FOR PAPER")
print("="*60)
print(f"Exp 1 (S^1):  Baseline circ={circ_base:.3f}  GCD circ={circ_gcd:.3f}")
print(f"Exp 2 (S^2):  Baseline H1={circ_hbase:.3f}   GCD H1={h1_hgcd:.3f} H2={h2_hgcd:.3f}")
print(f"Exp 3 (T^2):  Baseline β1={torus_dbase}        GCD β1={torus_dgcd}")
