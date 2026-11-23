from pathlib import Path
import sys
import pandas as pd
import numpy as np
from scipy.stats import t
d=pd.read_csv('base_municipal.csv')
reconstruir = '--reconstruir' in sys.argv or not Path('favorecidos_out2021.csv').exists()
if not reconstruir:
    c=pd.read_csv('favorecidos_out2021.csv')
else:
    s=pd.read_csv('siafi.csv',header=None,encoding='latin1',sep=';',usecols=[0,4]).rename(columns={0:'siafi',4:'codigo_ibge'})
    c=pd.read_csv('nis21_ag.csv').merge(s,on='siafi',validate='one_to_one')
d=d.merge(c[['codigo_ibge','nis21']],validate='one_to_one')
assert len(d)==5570 and d.nis21.sum()==14654782 and d.nis.sum()==20065780
print('Total nacional por município',d.nis21.sum(),d.nis.sum())
if reconstruir:
    d[['codigo_ibge','nis21']].to_csv('favorecidos_out2021.csv',index=False)
d=d[d.uf!='DF'].copy()
base21=1000*d.nis21/d['pop']
delta=d.fav-base21
uf=pd.get_dummies(d.uf,drop_first=True,dtype=float)
Y=100*d.pt2022.to_numpy()
rows=[]
for nome,terms in [('UF',[uf]),('UF e Haddad',[uf,100*d.pt2018]),('UF, Haddad e nível inicial',[uf,100*d.pt2018,base21/10])]:
 X=np.column_stack([np.ones(len(d)),delta/10,*terms])
 b=np.linalg.lstsq(X,Y,rcond=None)[0];e=Y-X@b
 A=np.linalg.inv(X.T@X)
 scores=pd.DataFrame(X*e[:,None]).groupby(d.uf.to_numpy()).sum().to_numpy()
 N,K=X.shape;G=len(scores)
 nomes = [('saldo',1)]
 if 'inicial' in nome: nomes.append(('inicial',len(b)-1))
 for termo,idx in nomes:
  ep=np.linalg.norm(scores@A[:,idx])*np.sqrt((N-1)/(N-K)*G/(G-1))
  margem=t.ppf(.975,G-1)*ep
  rows.append([nome,termo,b[idx],ep,b[idx]-margem,b[idx]+margem,N,G])
res=pd.DataFrame(rows,columns=['modelo','termo','estimativa','ep','lo','hi','n','grupos']);res.to_csv('expansao.csv',index=False)
print('delta cobertura média/mediana',delta.mean(),delta.median()); print(res)
