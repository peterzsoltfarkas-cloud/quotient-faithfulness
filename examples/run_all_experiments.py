"""
GCD Training Experiments v2 — tuned hyperparameters for Phase Cylinder
"""
import torch, math, numpy as np
import torch.nn as nn, torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42); np.random.seed(42)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, layers=3):
        super().__init__()
        net = []
        dims = [in_dim]+[hidden]*(layers-1)+[out_dim]
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
        comms=[A[i]@A[j]-A[j]@A[i] for i in range(self.k) for j in range(i+1,self.k)]
        if not comms: return torch.tensor(0.)
        return torch.stack(comms).pow(2).mean()

def eq_loss(enc,gen,x,x_aug,c):
    return nn.functional.mse_loss(enc(x_aug),gen.action(enc(x),c))

def smooth(v,w=20):
    return np.convolve(v,np.ones(w)/w,mode='valid')

# ── Exp 1: Phase Cylinder ─────────────────────────────────────────────────────
print("Exp 1: Phase Cylinder...")

def pc_batch(B=256):
    theta=torch.rand(B)*2*math.pi
    x=torch.stack([torch.cos(theta),torch.sin(theta)],1)+0.05*torch.randn(B,2)
    delta=(torch.rand(B)-0.5)*0.5
    xaug=torch.stack([torch.cos(theta+delta),torch.sin(theta+delta)],1)+0.05*torch.randn(B,2)
    return x,xaug,delta.unsqueeze(1),theta.unsqueeze(1)

h1={'task':[],'eq':[],'cl':[],'base':[]}
eb=MLP(2,1); hb=nn.Linear(1,1)
ob=optim.Adam(list(eb.parameters())+list(hb.parameters()),lr=1e-3)
eg=MLP(2,2); gg=LieGen(2,1); hg=nn.Linear(2,1)
og=optim.Adam(list(eg.parameters())+list(gg.parameters())+list(hg.parameters()),lr=5e-4)

for ep in range(500):
    x,xa,d,y=pc_batch()
    lt=((hb(eb(x))-y)**2).mean()
    ob.zero_grad();lt.backward();ob.step()
    h1['base'].append(lt.item())
    z=eg(x)
    lt=((hg(z)-y)**2).mean()
    le=eq_loss(eg,gg,x,xa,d)
    lc=gg.closure_loss()
    loss=lt+0.5*le+0.05*lc
    og.zero_grad();loss.backward();og.step()
    h1['task'].append(lt.item()); h1['eq'].append(le.item()); h1['cl'].append(lc.item())
print(f"  task:{h1['task'][-1]:.4f} eq:{h1['eq'][-1]:.4f} base:{h1['base'][-1]:.4f}")

# ── Exp 2: Hopf Fibration ─────────────────────────────────────────────────────
print("Exp 2: Hopf Fibration...")

def hopf(x):
    z1r,z1i,z2r,z2i=x[:,0],x[:,1],x[:,2],x[:,3]
    return torch.stack([2*(z1r*z2r+z1i*z2i),2*(z1i*z2r-z1r*z2i),
                        z1r**2+z1i**2-z2r**2-z2i**2],1)

def hf_batch(B=256):
    x=torch.randn(B,4); x=x/x.norm(dim=1,keepdim=True)
    t=torch.rand(B)*0.3
    c,s=torch.cos(t),torch.sin(t)
    xa=torch.stack([c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1],
                    c*x[:,2]-s*x[:,3],s*x[:,2]+c*x[:,3]],1)
    return x,xa,t.unsqueeze(1),hopf(x)

h2={'task':[],'eq':[],'cl':[],'base':[]}
ebh=MLP(4,2,hidden=128); hbh=nn.Linear(2,3)
obh=optim.Adam(list(ebh.parameters())+list(hbh.parameters()),lr=1e-3)
egh=MLP(4,3,hidden=128); ggh=LieGen(3,1); hgh=nn.Linear(3,3)
ogh=optim.Adam(list(egh.parameters())+list(ggh.parameters())+list(hgh.parameters()),lr=1e-3)

for ep in range(500):
    x,xa,t,y=hf_batch()
    lt=((hbh(ebh(x))-y)**2).mean()
    obh.zero_grad();lt.backward();obh.step()
    h2['base'].append(lt.item())
    z=egh(x)
    lt=((hgh(z)-y)**2).mean()
    le=eq_loss(egh,ggh,x,xa,t)
    lc=ggh.closure_loss()
    loss=lt+2.0*le+0.2*lc
    ogh.zero_grad();loss.backward();ogh.step()
    h2['task'].append(lt.item()); h2['eq'].append(le.item()); h2['cl'].append(lc.item())
print(f"  task:{h2['task'][-1]:.5f} eq:{h2['eq'][-1]:.5f} base:{h2['base'][-1]:.4f}")

# ── Exp 3: Double Pendulum ────────────────────────────────────────────────────
print("Exp 3: Double Pendulum...")

def dp_deriv(s,g=9.8):
    th1,th2,w1,w2=s; d=th2-th1; dm=2-np.cos(2*d)
    dw1=(-g*(2*np.sin(th1)+np.sin(th1-2*th2))-2*np.sin(d)*(w2**2+w1**2*np.cos(d)))/dm
    dw2=(2*np.sin(d)*(2*w1**2+2*g*np.cos(th1)+w2**2*np.cos(d)))/dm
    return np.array([w1,w2,dw1,dw2])

def rk4(s,dt=0.05):
    k1=dp_deriv(s);k2=dp_deriv(s+0.5*dt*k1)
    k3=dp_deriv(s+0.5*dt*k2);k4=dp_deriv(s+dt*k3)
    return s+(dt/6)*(k1+2*k2+2*k3+k4)

data=[]
for _ in range(60):
    s=np.array([np.random.uniform(-math.pi,math.pi),np.random.uniform(-math.pi,math.pi),
                np.random.uniform(-1,1),np.random.uniform(-1,1)])
    for _ in range(120): s=rk4(s); data.append(s.copy())
data=np.array(data,dtype=np.float32)

def enc_dp(d):
    th1,th2,w1,w2=d[:,0],d[:,1],d[:,2],d[:,3]
    return np.stack([np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),w1,w2],-1)

Xdp=torch.tensor(enc_dp(data))

def dp_batch(B=256):
    idx=torch.randint(len(Xdp),(B,)); x=Xdp[idx]
    d1=(torch.rand(B)-0.5)*0.4; d2=(torch.rand(B)-0.5)*0.4
    c1,s1,c2,s2=torch.cos(d1),torch.sin(d1),torch.cos(d2),torch.sin(d2)
    x0=c1*x[:,0]-s1*x[:,1]; x1=s1*x[:,0]+c1*x[:,1]
    x2=c2*x[:,2]-s2*x[:,3]; x3=s2*x[:,2]+c2*x[:,3]
    return x,torch.stack([x0,x1,x2,x3,x[:,4],x[:,5]],1),torch.stack([d1,d2],1)

h3={'task':[],'eq':[],'cl':[],'base':[]}
ebd=MLP(6,2,hidden=128); hbd=nn.Linear(2,2)
obd=optim.Adam(list(ebd.parameters())+list(hbd.parameters()),lr=1e-3)
egd=MLP(6,3,hidden=128); ggd=LieGen(3,2); hgd=nn.Linear(3,2)
ogd=optim.Adam(list(egd.parameters())+list(ggd.parameters())+list(hgd.parameters()),lr=1e-3)

for ep in range(500):
    x,xa,c=dp_batch()
    lt=((hbd(ebd(x))-x[:,4:6])**2).mean()
    obd.zero_grad();lt.backward();obd.step()
    h3['base'].append(lt.item())
    z=egd(x)
    lt=((hgd(z)-x[:,4:6])**2).mean()
    le=eq_loss(egd,ggd,x,xa,c)
    lc=ggd.closure_loss()
    loss=lt+1.5*le+0.1*lc
    ogd.zero_grad();loss.backward();ogd.step()
    h3['task'].append(lt.item()); h3['eq'].append(le.item()); h3['cl'].append(lc.item())
print(f"  task:{h3['task'][-1]:.5f} eq:{h3['eq'][-1]:.6f} cl:{h3['cl'][-1]:.2e} base:{h3['base'][-1]:.5f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig,axes=plt.subplots(2,3,figsize=(15,8))
fig.suptitle('GCD Training Dynamics — Farkas (2026), Papers II–VI',
             fontsize=13,fontweight='bold')

Cs={'task':'#d62728','eq':'#1f77b4','cl':'#2ca02c','base':'#888888'}

exps=[
    ('Phase Cylinder\n$Q=S^1$,  $\\mathrm{emb}(Q)=2$',
     h1,500,'Paper I–V running example\nbaseline: m=1, task loss only'),
    ('Hopf Fibration\n$Q=S^2$,  $\\mathrm{emb}(Q)=3$',
     h2,500,'Paper V §3.3\nbaseline: m=2, dimension trap'),
    ('Double Pendulum\n$Q=T^2$,  $\\mathrm{emb}(Q)=3$',
     h3,500,'vs. HNN baseline (Greydanus 2019)\nbaseline: m=2, no structural prior'),
]

for col,(title,h,ep,sub) in enumerate(exps):
    epv=np.arange(ep); sw=20; eps=epv[sw-1:]

    ax=axes[0,col]
    ax.semilogy(eps,smooth(h['task'],sw),color=Cs['task'],lw=2,label='Task loss (GCD)')
    ax.semilogy(eps,smooth(h['eq'],sw),  color=Cs['eq'],  lw=2,label='Equivariance loss')
    if max(h['cl'])>1e-14:
        ax.semilogy(eps,smooth(h['cl'],sw),color=Cs['cl'],lw=2,label='Closure $\\|[A,A]\\|^2$')
    ax.set_title(f'{title}\n{sub}',fontsize=8.5,fontweight='bold')
    ax.set_xlabel('Epoch',fontsize=9); ax.set_ylabel('Loss',fontsize=9)
    ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    ax2=axes[1,col]
    ax2.semilogy(eps,smooth(h['base'],sw),color=Cs['base'],lw=2,ls='--',
                 label='Baseline (no structural prior)')
    ax2.semilogy(eps,smooth(h['task'],sw),color=Cs['task'],lw=2,label='GCD')
    fgcd=smooth(h['task'],sw)[-1]; fbase=smooth(h['base'],sw)[-1]
    ax2.set_title('Task loss: GCD vs baseline',fontsize=9)
    ax2.set_xlabel('Epoch',fontsize=9); ax2.set_ylabel('Task loss',fontsize=9)
    ax2.legend(fontsize=8); ax2.grid(True,alpha=0.3)
    ax2.text(0.97,0.97,f'GCD:  {fgcd:.5f}\nBase: {fbase:.5f}',
             transform=ax2.transAxes,fontsize=7.5,va='top',ha='right',
             bbox=dict(boxstyle='round',facecolor='white',alpha=0.85))

plt.tight_layout()
plt.savefig('/home/claude/gcd_loss_curves.png',dpi=200,bbox_inches='tight')
print("Saved gcd_loss_curves.png")
