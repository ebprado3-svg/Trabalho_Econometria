import pandas as pd,numpy as np
a=[]
cols=['MÊS COMPETÊNCIA','CÓDIGO MUNICÍPIO SIAFI','NIS FAVORECIDO']
for chunk in pd.read_csv('nis.zip',usecols=cols,encoding='latin1',sep=';',chunksize=500000):
 chunk=chunk.loc[chunk[cols[0]]==202209].dropna()
 a.append((chunk[cols[1]].astype('int64')*10**11 + chunk[cols[2]].astype('int64')).to_numpy())
u=np.unique(np.concatenate(a))//10**11
codes,counts=np.unique(u,return_counts=True)
pd.DataFrame({'siafi':codes,'nis':counts}).to_csv('nis_ag.csv',index=False)
print(len(codes),counts.sum())
