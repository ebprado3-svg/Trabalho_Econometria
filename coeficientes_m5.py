"""Contrastes das covariáveis na especificação M5 da folha, peso municipal igual."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

pasta=Path(__file__).resolve().parent
d=pd.read_csv(pasta/'base_municipal.csv').query("uf != 'DF'").sort_values('codigo_ibge')
uf=pd.get_dummies(d.uf,drop_first=True,dtype=float)
X=np.column_stack([np.ones(len(d)), d.folha/10, uf, 100*d.pt2018, np.log(d.renda), d.edu, d.urb, d.jovens, d.idosos, np.log(d['pop'])])
b=np.linalg.lstsq(X,100*d.pt2022,rcond=None)[0]
res=100*d.pt2022.to_numpy()-X@b
A=np.linalg.inv(X.T@X)
grupos=pd.DataFrame(X*res[:,None]).groupby(d.uf.to_numpy()).sum().to_numpy()
n,k=X.shape; g=len(grupos)
ep=np.sqrt(np.diag(A @ grupos.T @ grupos @ A.T)*(g/(g-1))*((n-1)/(n-k)))
q=t.ppf(.975,g-1)
nomes=['Haddad 2018 (+1 p.p.)','Renda (+10%)','Superior completo (+1 p.p.)','Urbana (+10 p.p.)','Menores de 18 (+10 p.p.)','65 ou mais (+10 p.p.)','População (+10%)']
fator=np.array([1,np.log(1.10),1,10,10,10,np.log(1.10)])
estimativa=b[-7:]*fator
df=pd.DataFrame({'contraste':nomes, 'estimativa_pp':estimativa, 'ic_inf':estimativa-q*ep[-7:]*fator,'ic_sup':estimativa+q*ep[-7:]*fator})
alvo=pd.read_csv(pasta/'progressao.csv').query("j==5 and teste=='igual'").iloc[0]
assert n==alvo.n and abs(alvo.b-b[1])<1e-8 and abs(alvo.gamma-b[-7])<1e-8
df.to_csv(pasta/'coeficientes_m5.csv',index=False)
print(df.round(3).to_string(index=False))
