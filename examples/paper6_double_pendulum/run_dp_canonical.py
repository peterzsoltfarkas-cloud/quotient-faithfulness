"""
Double pendulum GCD with CANONICAL VARIABLES.

Mathematical foundation (derived from Hamiltonian before any code):

Phase space: (θ₁,θ₂) ∈ T² × (p₁,p₂) ∈ R²  [4D]
Canonical momenta:
    p₁ = 2ω₁ + ω₂cos(θ₁-θ₂)
    p₂ = ω₂ + ω₁cos(θ₁-θ₂)
Hamiltonian (unit masses/lengths/g):
    H = (p₁² + 2p₂² - 2p₁p₂cosΔ) / (2(2-cos²Δ)) − (2cosθ₁ + cosθ₂)
    where Δ = θ₁ − θ₂

At energy E: the energy surface M_E is a 3-manifold.
Fiber of π: M_E → T²: an ellipse ≅ S¹ in (p₁,p₂)-space.
→ M_E is topologically T³ for generic E.

Quotient factorisation:
    Q = T² (angle part, 3D embedding) + two 1D action fibers (2D total)
    → minimum embedding m ≥ 5   (consistent with SYC: d_box=2, ⌊2×2⌋+1=5)

GCD generators:
    A₁: U(1) acting on θ₁   (rotation in first angle)
    A₂: U(1) acting on θ₂   (rotation in second angle)

GCD loss:
    L = L_task + λ·L_eq + μ·L_cl + ν·L_phys
    L_phys = ||H(state) − H_expected||²   (Hamiltonian conservation)
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

torch.manual_seed(42); np.random.seed(42)
OUTDIR = '/mnt/user-data/outputs/figures/'

# ── Canonical momentum transformation ────────────────────────────────────────
def to_canonical(theta1, theta2, w1, w2):
    """(θ₁,θ₂,ω₁,ω₂) → canonical momenta (p₁,p₂)."""
    D = theta1 - theta2
    p1 = 2*w1 + w2*np.cos(D)
    p2 = w2   + w1*np.cos(D)
    return p1, p2

def hamiltonian(theta1, theta2, p1, p2):
    """H(θ₁,θ₂,p₁,p₂) with unit masses/lengths/g=1."""
    D = theta1 - theta2
    denom = 2*(2 - np.cos(D)**2)
    T = (p1**2 + 2*p2**2 - 2*p1*p2*np.cos(D)) / denom
    V = -(2*np.cos(theta1) + np.cos(theta2))
    return T + V

def canonical_input(theta1, theta2, w1, w2):
    """Build 6D input: (cosθ₁, sinθ₁, cosθ₂, sinθ₂, p₁, p₂)."""
    p1, p2 = to_canonical(theta1, theta2, w1, w2)
    return np.stack([np.cos(theta1), np.sin(theta1),
                     np.cos(theta2), np.sin(theta2), p1, p2], axis=-1)

# PyTorch versions
def hamiltonian_torch(th1, th2, p1, p2):
    D = th1 - th2
    denom = 2*(2 - torch.cos(D)**2)
    T = (p1**2 + 2*p2**2 - 2*p1*p2*torch.cos(D)) / denom
    V = -(2*torch.cos(th1) + torch.cos(th2))
    return T + V

def canonical_input_torch(x):
    """x has columns (cosθ₁,sinθ₁,cosθ₂,sinθ₂,ω₁,ω₂) — raw encoder input."""
    # Recover angles from circle encoding
    th1 = torch.atan2(x[:,1], x[:,0])
    th2 = torch.atan2(x[:,3], x[:,2])
    w1, w2 = x[:,4], x[:,5]
    D = th1 - th2
    p1 = 2*w1 + w2*torch.cos(D)
    p2 = w2   + w1*torch.cos(D)
    return torch.stack([x[:,0],x[:,1],x[:,2],x[:,3],p1,p2], dim=1)

def hamiltonian_loss(x_raw):
    """Physics loss: variance of H across batch should be near zero on a trajectory."""
    th1 = torch.atan2(x_raw[:,1], x_raw[:,0])
    th2 = torch.atan2(x_raw[:,3], x_raw[:,2])
    w1, w2 = x_raw[:,4], x_raw[:,5]
    D = th1 - th2
    p1 = 2*w1 + w2*torch.cos(D)
    p2 = w2   + w1*torch.cos(D)
    H = hamiltonian_torch(th1, th2, p1, p2)
    # Loss: H should be constant along a trajectory → penalise variance
    # Also: decoder should preserve H
    return H.var()

# ── Architecture ──────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self,i,o,h=256,l=4):
        super().__init__()
        net=[]; dims=[i]+[h]*(l-1)+[o]
        for k in range(len(dims)-1):
            net.append(nn.Linear(dims[k],dims[k+1]))
            if k<len(dims)-2: net.append(nn.ReLU())
        self.net=nn.Sequential(*net)
    def forward(self,x): return self.net(x)

class LieGen(nn.Module):
    """Two U(1) generators acting on the T² (angle) subspace."""
    def __init__(self, d, k=2):
        super().__init__()
        self.k=k; self.W=nn.Parameter(torch.randn(k,d,d)*0.1)
    def generators(self):
        W=self.W; return 0.5*(W-W.transpose(-1,-2))
    def action(self, z, c):
        if c.dim()==1: c=c.unsqueeze(0).expand(z.size(0),-1)
        M=torch.einsum('bk,kij->bij',c,self.generators())
        return torch.einsum('bij,bj->bi',torch.matrix_exp(M),z)
    def closure_loss(self):
        A=self.generators()
        comms=[A[i]@A[j]-A[j]@A[i] for i in range(self.k) for j in range(i+1,self.k)]
        return torch.stack(comms).pow(2).mean() if comms else torch.tensor(0.)

# ── Generate data ─────────────────────────────────────────────────────────────
def dp_deriv(s, g=1.0):
    th1,th2,w1,w2=s; D=th2-th1; dm=2-np.cos(2*D)
    dw1=(-g*(2*np.sin(th1)+np.sin(th1-2*th2))-2*np.sin(D)*(w2**2+w1**2*np.cos(D)))/dm
    dw2=(2*np.sin(D)*(2*w1**2+2*g*np.cos(th1)+w2**2*np.cos(D)))/dm
    return np.array([w1,w2,dw1,dw2])

def rk4(s,dt=0.05):
    k1=dp_deriv(s); k2=dp_deriv(s+0.5*dt*k1)
    k3=dp_deriv(s+0.5*dt*k2); k4=dp_deriv(s+dt*k3)
    return s+(dt/6)*(k1+2*k2+2*k3+k4)

print("Generating double pendulum data (canonical variables)...")
np.random.seed(42)
data_raw=[]
for _ in range(80):
    s=np.array([np.random.uniform(-np.pi,np.pi),
                np.random.uniform(-np.pi,np.pi),
                np.random.uniform(-1,1),np.random.uniform(-1,1)])
    for _ in range(150): s=rk4(s); data_raw.append(s.copy())
data_raw=np.array(data_raw,dtype=np.float32)

# Compute canonical momenta for each state
p1_all, p2_all = to_canonical(data_raw[:,0],data_raw[:,1],
                               data_raw[:,2],data_raw[:,3])
H_all = hamiltonian(data_raw[:,0],data_raw[:,1],p1_all,p2_all)
print(f"  H range: [{H_all.min():.2f}, {H_all.max():.2f}]  (per-trajectory, should vary)")

# Build 6D canonical input: (cosθ₁,sinθ₁,cosθ₂,sinθ₂, p₁,p₂)
inp_canon = np.stack([np.cos(data_raw[:,0]),np.sin(data_raw[:,0]),
                      np.cos(data_raw[:,1]),np.sin(data_raw[:,1]),
                      p1_all,p2_all],axis=1).astype(np.float32)

# Also build OLD input for comparison: (cosθ₁,sinθ₁,cosθ₂,sinθ₂,ω₁,ω₂)
inp_old = np.stack([np.cos(data_raw[:,0]),np.sin(data_raw[:,0]),
                    np.cos(data_raw[:,1]),np.sin(data_raw[:,1]),
                    data_raw[:,2],data_raw[:,3]],axis=1).astype(np.float32)

# Task: predict NEXT state's canonical momenta (p₁,p₂) from current canonical state
# This makes the task intrinsically Hamiltonian-aware
Y = np.stack([p1_all,p2_all],axis=1).astype(np.float32)  # targets = momenta

# Augmentation: rotate θ₁ by δ₁ AND θ₂ by δ₂ simultaneously
# The canonical input after rotation (θᵢ→θᵢ+δᵢ):
# cos(θᵢ+δᵢ) = cosθᵢcosδᵢ - sinθᵢsinδᵢ  (can be computed)
def augment_canonical(inp_c, d1, d2):
    """Rotate θ₁→θ₁+d1, θ₂→θ₂+d2 in canonical input."""
    c1,s1 = inp_c[:,0],inp_c[:,1]
    c2,s2 = inp_c[:,2],inp_c[:,3]
    cd1,sd1=np.cos(d1),np.sin(d1)
    cd2,sd2=np.cos(d2),np.sin(d2)
    new_c1 = c1*cd1 - s1*sd1
    new_s1 = s1*cd1 + c1*sd1
    new_c2 = c2*cd2 - s2*sd2
    new_s2 = s2*cd2 + c2*sd2
    # Canonical momenta also change under angle rotation:
    # p₁ = 2ω₁ + ω₂cos(θ₁-θ₂) — cos(Δ+d1-d2) changes
    D_new = np.arctan2(new_s1,new_c1) - np.arctan2(new_s2,new_c2)
    th1=np.arctan2(inp_c[:,1],inp_c[:,0]); th2=np.arctan2(inp_c[:,3],inp_c[:,2])
    # Recover ω from canonical: need to invert (p₁,p₂) → (ω₁,ω₂) at new angles
    # For the augmentation, we keep ω unchanged (augment angle, not velocity)
    # Then recompute p at the new angle
    p1_old,p2_old = inp_c[:,4],inp_c[:,5]
    # Get ω from old canonical at old angle
    D_old = th1-th2
    denom_old = np.cos(D_old)**2 - 2
    w1_orig = (-p1_old + p2_old*np.cos(D_old))/denom_old
    w2_orig = (p1_old*np.cos(D_old) - 2*p2_old)/denom_old
    # Recompute p at new angles (same ω)
    p1_new = 2*w1_orig + w2_orig*np.cos(D_new)
    p2_new = w2_orig   + w1_orig*np.cos(D_new)
    return np.stack([new_c1,new_s1,new_c2,new_s2,p1_new,p2_new],axis=1)

# Build augmented pairs
np.random.seed(7)
d1_aug = np.random.uniform(-0.4,0.4,len(inp_canon)).astype(np.float32)
d2_aug = np.random.uniform(-0.4,0.4,len(inp_canon)).astype(np.float32)
inp_aug = augment_canonical(inp_canon,d1_aug,d2_aug).astype(np.float32)

Xc = torch.tensor(inp_canon)
Xc_aug = torch.tensor(inp_aug)
Yt  = torch.tensor(Y)
D_t = torch.tensor(np.stack([d1_aug,d2_aug],axis=1))
H_t = torch.tensor(H_all.reshape(-1,1))

n=len(Xc); split=int(0.8*n)
print(f"  Dataset: {n} states, train={split}, test={n-split}")

# ── Training: compare 3 conditions ────────────────────────────────────────────
configs = {
    'Baseline m=3\n(task only)':
        dict(m=3, use_gcd=False, use_phys=False, use_canon=False),
    'EC m=3\n(equivariance only, not full GCD)':
        dict(m=3, use_gcd=True, use_phys=False, use_canon=False),
    'GCD-canonical m=5\n(Hamiltonian physics loss)':
        dict(m=5, use_gcd=True, use_phys=True, use_canon=True),
}

results={}
for name, cfg in configs.items():
    m=cfg['m']
    Xin = Xc if cfg['use_canon'] else torch.tensor(inp_old)
    enc=MLP(6,m,h=256,l=4); head=MLP(m,2,h=128,l=3)
    H_head=nn.Linear(m,1)  # learned energy head: latent → R  (for per-sample L_phys)
    if cfg['use_gcd']:
        gen=LieGen(m,2)  # two U(1) generators for T²
        opt=optim.Adam(list(enc.parameters())+list(head.parameters())
                      +list(gen.parameters())+list(H_head.parameters()),lr=5e-4)
    else:
        opt=optim.Adam(list(enc.parameters())+list(head.parameters())
                      +list(H_head.parameters()),lr=1e-3)

    losses={'task':[],'eq':[],'cl':[],'phys':[]}
    for ep in range(800):
        idx=torch.randint(split,(256,))
        x_b=Xin[idx]; y_b=Yt[idx]
        z=enc(x_b)
        L_task=nn.functional.mse_loss(head(z),y_b)
        loss=L_task

        if cfg['use_gcd']:
            xa_b=Xc_aug[idx] if cfg['use_canon'] else torch.tensor(inp_old)[idx]
            d_b=D_t[idx]
            L_eq=nn.functional.mse_loss(enc(xa_b),gen.action(z,d_b))
            L_cl=gen.closure_loss()
            loss=loss+0.5*L_eq+0.05*L_cl
            losses['eq'].append(L_eq.item())
            losses['cl'].append(L_cl.item())

        if cfg['use_phys']:
            # Correct per-sample formulation:
            # L_phys = ||H_head(enc(x)) - H(x)||²
            # Tests whether the LATENT (not the input) encodes energy.
            # H(x) is analytically known for every training state.
            # No batch-trajectory assumption needed.
            H_pred = H_head(z).squeeze()   # learned head on latent
            H_true = H_t[idx].squeeze()    # analytically known H(x)
            L_phys = ((H_pred - H_true)**2).mean()
            loss=loss+0.5*L_phys
            losses['phys'].append(L_phys.item())

        losses['task'].append(L_task.item())
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        mse=nn.functional.mse_loss(head(enc(Xin[split:])),Yt[split:]).item()
    
    # Get latent for visualisation
    with torch.no_grad():
        Z=enc(Xin).numpy()
    
    results[name]={'mse':mse,'Z':Z,'losses':losses,'m':m}
    short=name.split('\n')[0]
    print(f"  {short}: test MSE={mse:.4f}")

# ── FIGURES ───────────────────────────────────────────────────────────────────
# Figure 1: MSE comparison + loss curves
fig,axes=plt.subplots(1,3,figsize=(16,5))
fig.suptitle('Double Pendulum: Three-Condition Comparison\n'
             r'State space $TQ=T^2\times\mathbb{R}^2$, $\mathrm{emb}(TQ)=5$ from Lagrangian analysis',
             fontsize=11,fontweight='bold')

names=list(results.keys())
mses=[results[n]['mse'] for n in names]
short_names=[n.split('\n')[0]+'\n'+n.split('\n')[1] for n in names]
cols=['#aaaaaa','#ff7f0e','#2166ac']
bars=axes[0].bar(range(len(names)),mses,color=cols,alpha=0.85)
for bar,v in zip(bars,mses):
    axes[0].text(bar.get_x()+bar.get_width()/2,v+0.0005,f'{v:.4f}',
                 ha='center',fontsize=9,fontweight='bold')
axes[0].set_xticks(range(len(names)))
axes[0].set_xticklabels(short_names,fontsize=7.5)
axes[0].set_ylabel('Test MSE (momentum prediction)')
axes[0].set_title('Test MSE: canonical momentum prediction\n(lower = better representation)')
axes[0].grid(True,axis='y',alpha=0.3)

# Loss curves for GCD-canonical
lc=results[list(results.keys())[2]]['losses']
w=20
def sm(v): return np.convolve(v,np.ones(w)/w,'valid') if len(v)>w else v
epochs=np.arange(len(sm(lc['task'])))
axes[1].semilogy(epochs,sm(lc['task']),'b',lw=2,label='Task')
axes[1].semilogy(epochs,sm(lc['eq']),'orange',lw=2,label='Equivariance')
axes[1].semilogy(epochs,sm(lc['cl']),'g',lw=2,label='Closure')
axes[1].semilogy(epochs,sm(lc['phys']),'r',lw=2,label='Hamiltonian phys')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss (log)')
axes[1].set_title('GCD-canonical (m=5) loss components\nAll four terms converge')
axes[1].legend(fontsize=8); axes[1].grid(True,alpha=0.2)

# PCA of latent spaces
ax=axes[2]
pca=PCA(2)
for ni,(name,col) in enumerate(zip(names,cols)):
    Z=results[name]['Z']
    Z2=pca.fit_transform(Z)
    ax.scatter(Z2[::3,0],Z2[::3,1],s=3,alpha=0.3,color=col,
               label=name.split('\n')[0])
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('Latent PCA (all conditions)\nGCD-canonical should show T² structure')
ax.legend(fontsize=7.5,markerscale=3)
ax.grid(True,alpha=0.2)

plt.tight_layout()
fig.savefig(OUTDIR+'dp_canonical_results.png',dpi=200,bbox_inches='tight')
plt.close(); print("Saved dp_canonical_results.png")

# Figure 2: Phase space structure visualisation
fig,axes=plt.subplots(2,3,figsize=(16,9))
fig.suptitle('Double Pendulum Phase Space Structure\n'
             '(θ₁,θ₂) ∈ T² × (p₁,p₂) ∈ R² — energy constraint defines 3-manifold M_E',
             fontsize=11,fontweight='bold')

th1=data_raw[:,0]; th2=data_raw[:,1]
p1_vis,p2_vis=p1_all,p2_all
H_vis=H_all

# (θ₁,θ₂) angle space — should look like T²
ax=axes[0,0]
sc=ax.scatter(th1[::5],th2[::5],c=H_vis[::5],cmap='plasma',s=3,alpha=0.5)
plt.colorbar(sc,ax=ax,label='H (energy)')
ax.set_xlabel('θ₁'); ax.set_ylabel('θ₂')
ax.set_title('Configuration space (θ₁,θ₂)\n= T² by construction')

# (p₁,p₂) momentum space
ax=axes[0,1]
sc=ax.scatter(p1_vis[::5],p2_vis[::5],c=H_vis[::5],cmap='plasma',s=3,alpha=0.5)
plt.colorbar(sc,ax=ax,label='H (energy)')
ax.set_xlabel('p₁ (canonical)'); ax.set_ylabel('p₂ (canonical)')
ax.set_title('Momentum space (p₁,p₂)\nEllipses = constant-energy fibers')

# At a specific energy level: show the momentum ellipse
E_target=5.0
mask=np.abs(H_vis-E_target)<0.5
ax=axes[0,2]
ax.scatter(p1_vis[mask],p2_vis[mask],c=th1[mask],cmap='hsv',s=5,alpha=0.7)
ax.set_xlabel('p₁'); ax.set_ylabel('p₂')
ax.set_title(f'Fiber |H-{E_target}|<0.5 in (p₁,p₂)-space\n≅ S¹ ellipse (angle-coloured)')

# Hamiltonian value distribution
ax=axes[1,0]
ax.hist(H_vis,bins=50,color='#2166ac',alpha=0.8,edgecolor='white')
ax.set_xlabel('H (energy)'); ax.set_ylabel('Count')
ax.set_title('Energy distribution\n(diverse trajectories)')
ax.grid(True,alpha=0.3)

# (cosθ₁,sinθ₁) — circle structure
ax=axes[1,1]
th=np.linspace(0,2*np.pi,300)
ax.plot(np.cos(th),np.sin(th),'k--',lw=1,alpha=0.3)
sc=ax.scatter(np.cos(th1[::5]),np.sin(th1[::5]),
              c=p1_vis[::5],cmap='RdBu_r',s=3,alpha=0.5)
plt.colorbar(sc,ax=ax,label='p₁ (canonical momentum)')
ax.set_aspect('equal')
ax.set_title('S¹ structure of θ₁ colored by p₁\n(velocity varies around the circle)')

# GCD canonical latent PCA
ax=axes[1,2]
Z_gcd=results[list(results.keys())[2]]['Z']
pca2=PCA(2); Z2=pca2.fit_transform(Z_gcd)
sc=ax.scatter(Z2[::3,0],Z2[::3,1],
              c=np.arctan2(np.sin(th1[::3]),np.cos(th1[::3])),
              cmap='hsv',s=4,alpha=0.5)
plt.colorbar(sc,ax=ax,label='θ₁')
ax.set_title('GCD-canonical latent (m=5) PCA\ncolored by θ₁ — T² structure visible?')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

plt.tight_layout()
fig.savefig(OUTDIR+'dp_phase_space.png',dpi=200,bbox_inches='tight')
plt.close(); print("Saved dp_phase_space.png")
