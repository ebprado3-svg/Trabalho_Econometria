"""Resultados da sequência central, seus diagnósticos e as curvas da figura 2."""
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
seed=202211
base=pd.read_csv('base_municipal.csv')
d=base.query("uf!='DF'").sort_values('codigo_ibge').merge(pd.read_csv('primeiro.csv'),validate='one_to_one').merge(pd.read_csv('coords.csv')[['codigo_ibge','latitude','longitude']],validate='one_to_one').reset_index(drop=True)
assert len(d)==5569
n=len(d)
x=d.folha.to_numpy()/10
y=d.pt2022.to_numpy()*100
h=d.pt2018.to_numpy()*100
uf=pd.get_dummies(d.uf,drop_first=True,dtype=float).to_numpy()
gr=pd.factorize(d.uf,sort=True)[0]
G=len(set(gr))
def design(j,exposure=x,history=h):
    cols=[np.ones(n),exposure]
    if j>=2: cols.extend(uf.T)
    if j>=3: cols.append(history)
    if j>=4: cols.extend([np.log(d.renda),d.edu])
    if j>=5: cols.extend([d.urb,d.jovens,d.idosos,np.log(d['pop'])])
    return np.column_stack(cols)
def fit(X,Y=y,w=None,g=gr):
    w=np.ones(len(Y)) if w is None else np.asarray(w)
    Xw=X*np.sqrt(w)[:,None]
    b=np.linalg.lstsq(Xw,Y*np.sqrt(w),rcond=None)[0]
    e=Y-X@b
    A=np.linalg.inv(Xw.T@Xw)
    scores=X*(w*e)[:,None]
    S=np.array([scores[g==k].sum(axis=0) for k in np.unique(g)])
    N,K=X.shape
    groups=len(S)
    c=groups/(groups-1)*(N-1)/(N-K)
    se=np.sqrt(np.diag((A@S.T)@(A@S.T).T)*c)
    r2=1-np.dot(w,e**2)/np.dot(w,(Y-np.average(Y,weights=w))**2)
    return b,e,A,se,r2,c
rows=[]
def record(j,label,X,Y=y,w=None):
    b,e,A,se,r2,c=fit(X,Y,w)
    q=t.ppf(.975,G-1)
    rows.append(dict(j=j,teste=label,b=b[1],lo=b[1]-q*se[1],hi=b[1]+q*se[1],r2=r2,gamma=b[2+uf.shape[1]] if j>=3 else np.nan,n=n))
    return b
for j in range(1,6):
    X=design(j)
    for label,w in [('igual',None),('população',d['pop']),('votos',d.validos2022)]:
        record(j,label,X,w=w)
    record(j,'NIS',design(j,d.fav.to_numpy()/10))
record(2,'antecedente',design(2),h)
record(2,'diferença',design(2),y-h)
record(3,'1º turno',design(3,history=d.haddad1.to_numpy()*100))
record(3,'H quadrático',np.column_stack([design(3),(h-h.mean())**2]))
record(3,'H por UF',np.column_stack([design(3),uf*(h-h.mean())[:,None]]))
for j in [3,5]:
    xx=np.column_stack([design(j),(x-x.mean())**2])
    b=record(j,'X quadrático',xx)
    print('inclinações da exposição nos quartis',j,b[1]+2*b[-1]*(np.quantile(x,[.25,.75])-x.mean()))
r=pd.DataFrame(rows)
r.to_csv('progressao.csv',index=False)
print(r.round(4).to_string(index=False))
q=r.set_index(['j','teste'])
assert np.isclose(q.loc[(2,'igual'),'b']-q.loc[(3,'igual'),'b'],q.loc[(2,'antecedente'),'b']*q.loc[(3,'igual'),'gamma'])
# Pares dirigidos: oito vizinhos municipais em distância de grande círculo.
lat,lon=np.deg2rad(d[['latitude','longitude']].values.T)
xyz=np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
neigh=cKDTree(xyz).query(xyz,k=9)[1]
i=np.repeat(np.arange(n),8); k=neigh[:,1:].ravel()
assert not (i==k).any()
keep=d.rgi.values[i]!=d.rgi.values[k]
diag=[]; dec=[]; omitidas=[]
for j in [2,3,4,5]:
    X=design(j);b,e,A,se,r2,c=fit(X)
    lever=np.sum((X@A)*X,axis=1)
    cook=e**2*lever/((1-lever)**2*X.shape[1]*(e@e/(n-X.shape[1])))
    retain=cook<=4/n
    bc=fit(X[retain],y[retain],g=gr[retain])[0][1]
    loo=[]
    for state in range(G):
        mask=gr!=state
        xx=X[mask]
        # Elimina a dummy vazia ou, se o grupo-base saiu, recentra as dummies.
        xx=np.column_stack([xx[:,:2],pd.get_dummies(d.uf[mask],drop_first=True,dtype=float),xx[:,2+uf.shape[1]:]])
        loo.append(fit(xx,y[mask],g=gr[mask])[0][1])
        omitidas.append((j,d.uf[gr==state].iloc[0],loo[-1],loo[-1]-b[1]))
    means=pd.Series(e).groupby(pd.qcut(x,10,labels=False)).mean().values
    dec.extend((j,g+1,v) for g,v in enumerate(means))
    z=e-e.mean()
    def moran(z,i,k): return n/len(i)*np.sum(z[i]*z[k])/(z@z)
    crossing=moran(z,i[keep],k[keep]); glob=moran(z,i,k)
    rng=np.random.default_rng(seed)
    vals=np.array([moran(rng.permutation(z),i[keep],k[keep]) for _ in range(999)])
    esperado=-1/(n-1)
    p=(1+sum(abs(vals-esperado)>=abs(crossing-esperado)))/1000
    diag.append([j,max(abs(means)),(~retain).sum(),bc,min(loo),max(loo),glob,crossing,p])
pd.DataFrame(omitidas,columns=['M','uf_omitida','b','diferenca']).to_csv('sem_uf.csv',index=False)
pd.DataFrame(dec,columns=['M','decil','residuo_pp']).to_csv('decis.csv',index=False)
D=pd.DataFrame(diag,columns=['M','max_abs','cook_N','sem_cook','min_sem_uf','max_sem_uf','moran','moran_entre','p_perm'])
D.to_csv('diagnosticos_centrais.csv',index=False)
print('arestas distintas RGI',keep.sum(),'de',len(i)); print(D.round(4))
# Wild cluster restricted bootstrap-t, Rademacher, CR1. Inverte H0: beta=c0.
def wild(j, peso, w):
    X=design(j)
    b,e,A,se,_,c=fit(X,w=w)
    w=np.ones(n) if w is None else np.asarray(w)
    signs=np.random.default_rng(seed).choice([-1.,1.],size=(G,9999))
    a=A[:,1];v=X@a
    def somas(z):
        return np.array([X[gr==g].T @ (w[gr==g]*z[gr==g]) for g in range(G)])
    S=somas(e); H=somas(v)
    def partes(s):
        delta=A@s.T@signs
        return delta[1],(s@a)[:,None]*signs-H@delta
    b0,s0=partes(S);b1,s1=partes(H/a[1])
    v0=c*(s0*s0).sum(axis=0);v1=2*c*(s0*s1).sum(axis=0);v2=c*(s1*s1).sum(axis=0)
    def pval(null):
        db=b[1]-null
        tstar=(b0+db*b1)/np.sqrt(v0+db*v1+db*db*v2)
        return (1+(abs(tstar)>=abs(db/se[1])).sum())/10000
    # Verificação direta de uma réplica sob H0, inclusive com pesos.
    br=b-a*b[1]/a[1];fitted=X@br
    ystar=fitted+(y-fitted)*signs[gr,0]
    bcheck,_,_,scheck,_,_=fit(X,ystar,w)
    assert np.allclose([bcheck[1],scheck[1]],[b0[0]+b[1]*b1[0],np.sqrt(v0[0]+b[1]*v1[0]+b[1]**2*v2[0])],atol=1e-8)
    ends=[]
    for sign in [-1,1]:
        inside=b[1];outside=inside+sign*8*se[1]
        assert pval(outside)<.05
        for _ in range(29):
            mid=(inside+outside)/2
            if pval(mid)>=.05: inside=mid
            else: outside=mid
        ends.append(inside)
    for delta in np.linspace(-12,12,145):
        pt=b[1]+se[1]*delta
        assert (ends[0]<=pt<=ends[1])==(pval(pt)>=.05)
    return [j,peso,*ends,pval(0)]
W=pd.DataFrame([wild(j,p,w) for j in range(1,6) for p,w in [('igual',None),('população',d['pop']),('votos',d.validos2022)]],columns=['M','peso','lo','hi','pzero'])
print(W.round(5));W.to_csv('wild.csv',index=False)
# Curvas para a análise descritiva com 5.570 municípios.
xx=base.folha.to_numpy(); yy=base.pt2022.to_numpy()
L=np.column_stack([np.ones(len(xx)),(xx-xx.mean())/100])
opt=minimize(lambda b:np.sum(np.logaddexp(0,L@b)-yy*(L@b)),np.zeros(2),jac=lambda b:L.T@(expit(L@b)-yy),method='BFGS')
grid=np.linspace(xx.min(),xx.max(),400)
plt.figure(figsize=(7.2,4.0))
plt.scatter(xx,100*yy,s=4,color='#63809b',alpha=.22,linewidths=0,rasterized=True)
for deg,label,color in [(1,'Linear','#222222'),(2,'Quadrático','#c65719')]:
    plt.plot(grid,np.polyval(np.polyfit(xx,100*yy,deg),grid),label=label,color=color,lw=1.8)
plt.plot(grid,100*expit(opt.x[0]+opt.x[1]*(grid-xx.mean())/100),color='#13755c',linestyle='--',label='Logit fracionário',lw=2)
plt.xlabel('Famílias na folha por mil habitantes');plt.ylabel('Votos válidos em Lula (%)')
plt.legend(frameon=False);plt.gca().spines[['top','right']].set_visible(False);plt.tight_layout()
plt.savefig('ajustes.png',dpi=220);plt.close()
# Permite inclinações estaduais da cobertura no modelo com Haddad em 2018.
X=design(3)
bh=fit(np.column_stack([X,uf*x[:,None]]))[0]
slopes=bh[1]+np.r_[0,bh[-uf.shape[1]:]]
pd.DataFrame({'uf':np.sort(d.uf.unique()),'b':slopes}).to_csv('inclinacoes_uf.csv',index=False)
print('inclinações estaduais: min, mediana, max',np.quantile(slopes,[0,.5,1]))
# Verificação direta de uma réplica do bootstrap H0: beta=0.
j=3;X=design(j);b,e,A,se,_,c=fit(X)
v=X@A[:,1]
restricted=b-A[:,1]*b[1]/A[1,1]
er=y-X@restricted
sign=np.random.default_rng(seed).choice([-1.,1.],size=G)
ys=X@restricted+er*sign[gr]
bb,_,_,sse,_,_=fit(X,ys)
S=np.array([X[gr==g].T@er[gr==g] for g in range(G)])
beta_star=A@S.T@sign
H=np.array([(v[gr==g,None]*X[gr==g]).sum(axis=0) for g in range(G)])
se_star=np.sqrt(c*np.sum(((S@A[:,1])*sign-H@beta_star)**2))
assert np.allclose([bb[1],sse[1]],[beta_star[1],se_star],rtol=1e-8)

from complementos import analisar
analisar(d, base, design, fit, gr, y, h, x, xyz, i, k)
