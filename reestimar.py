"""Reprodução com base_municipal.csv. Dependências: pandas, numpy, scipy."""
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.special import expit
from scipy.optimize import minimize
from pathlib import Path

d=pd.read_csv(Path(__file__).with_name('base_municipal.csv'))
d['ln_renda']=np.log(d.renda)
d['ln_pop']=np.log(d['pop'])
d=d.sort_values('codigo_ibge').reset_index(drop=True)
amostra=d[d.uf!='DF'].copy()
rows=[]

def ajustar(base, termo, ajustes=(), uf=False, grupo='uf', peso=None, nome=''):
    X=base[[termo,*ajustes]].to_numpy(float)
    if uf:
        X=np.column_stack([X,pd.get_dummies(base.uf,drop_first=True,dtype=float)])
    X=np.column_stack([np.ones(len(X)),X])
    y=base.pt2022.to_numpy()
    w=np.ones(len(X)) if peso is None else base[peso].to_numpy()
    xw=X*np.sqrt(w)[:,None]
    b=np.linalg.lstsq(xw,y*np.sqrt(w),rcond=None)[0]
    resid=y-X@b
    A=np.linalg.inv(xw.T@xw)
    scores=X*(w*resid)[:,None]
    sums=pd.DataFrame(scores).groupby(base[grupo].to_numpy()).sum().to_numpy()
    G=len(sums);n,k=X.shape
    cov=A@sums.T@sums@A*((n-1)/(n-k)*G/(G-1))
    ep=np.sqrt(np.diag(cov))
    ci=t.ppf(.975,G-1)*ep[1]
    rows.append(dict(modelo=nome,n=n,G=G,b=1000*b[1],ep=1000*ep[1],lo=1000*(b[1]-ci),hi=1000*(b[1]+ci)))
    h=np.sum((xw@A)*xw,axis=1)
    cook=w*resid**2*h/(k*(np.sum(w*resid**2)/(n-k))*(1-h)**2)
    return b,ep,cook

controls=['renda','edu','urb','jovens','idosos','ln_pop']
for x in ['folha','fav']:
    for lab,ad,fe in [('simples',[],False),('UF',[],True),('UF+2018',['pt2018'],True),('UF+2018+soc',['pt2018','edu','ln_renda'],True)]:
        ajustar(amostra,x,ad,fe,nome=x+' '+lab)
    ajustar(amostra,x,controls[:2],True,nome=x+' UF + renda em nível e escolaridade (histórico)')
    b,ep,cook=ajustar(amostra,x,controls,True,nome=x+' UF + bloco demográfico e renda em nível (histórico)')
    ajustar(amostra.loc[cook<=4/len(amostra)],x,controls,True,nome=x+' Cook')
    for var in ['jovens','idosos']:
        ajustar(amostra,x,[v for v in controls if v!=var],True,nome=x+' sem '+var)
    for w in ['pop','validos2022']:
        ajustar(amostra,x,controls,True,peso=w,nome=x+' peso '+w)
for grp in ['uf','rgi']:
    ajustar(d,'folha',grupo=grp,nome='referencia '+grp)
    b,ep,_=ajustar(d,'folha',['edu','ln_renda'],grupo=grp,nome='equacao3 '+grp)
    print('renda',grp,b[-1],100*np.log(1.1)*b[-1],100*.1*b[-1])
ajustar(d[d.familias<=d.dom],'folha',grupo='rgi',nome='sem excesso')
ex=d.familias>d.dom
print('excesso',ex.sum(),'ultimo decil',(ex & (d.folha>=d.folha.quantile(.9))).sum())
print('corr folha NIS',amostra.folha.corr(amostra.fav))
print('10,90% e quartis',d.folha.quantile([.1,.25,.75,.9]))
# Modelo logit fracionário de referência, cada município tem peso igual.
x=np.column_stack([np.ones(len(d)),(d.folha-d.folha.mean())/100])
y=d.pt2022.to_numpy()
opt=minimize(lambda v:np.sum(np.logaddexp(0,x@v)-y*(x@v)),np.zeros(2),jac=lambda v:x.T@(expit(x@v)-y),method='BFGS',options={'gtol':1e-6})
assert np.max(np.abs(opt.jac))<1e-4
q=d.folha.quantile([.25,.75]).to_numpy()
print('logit fracionario: Q1/Q3',100*expit(opt.x[0]+opt.x[1]*(q-d.folha.mean())/100))
res=pd.DataFrame(rows)
print(res.round(4).to_string(index=False))
res.to_csv(Path(__file__).with_name('estimativas.csv'),index=False)
z=(d.folha-d.folha.mean())/10
B=np.linalg.lstsq(np.column_stack((np.ones(len(d)),z,z*z)),100*d.pt2022,rcond=None)[0]
zq=(q-d.folha.mean())/10
print('quadratica',B,'dif',np.diff(B[1]*zq+B[2]*zq*zq),'derivadas',B[1]+2*B[2]*zq)

from referencia import calcular
calcular(d)
