import pandas as pd
from zipfile import ZipFile
with ZipFile('tse2018.zip') as z:
 f=next(n for n in z.namelist() if n.endswith('_BR.csv'))
 a=pd.read_csv(z.open(f),sep=';',encoding='latin1',usecols=['DS_CARGO','NR_TURNO','SG_UF','CD_MUNICIPIO','QT_VOTOS_NOMINAIS','NR_CANDIDATO'])
a=a[(a.DS_CARGO.str.upper()=='PRESIDENTE')&(a.NR_TURNO==1)&(a.SG_UF!='ZZ')]
b=a.groupby(['SG_UF','CD_MUNICIPIO','NR_CANDIDATO']).QT_VOTOS_NOMINAIS.sum().unstack(fill_value=0)
out=(b[13]/b.sum(axis=1)).rename('haddad1').reset_index()
with ZipFile('tsecross.zip') as z: c=pd.read_csv(z.open('municipio_tse_ibge.csv'),sep=';',encoding='latin1')
out=out.merge(c,left_on=['SG_UF','CD_MUNICIPIO'],right_on=['SG_UF','CD_MUNICIPIO_TSE'],validate='one_to_one')
assert len(out)==5570
out.rename(columns={'CD_MUNICIPIO_IBGE':'codigo_ibge'})[['codigo_ibge','haddad1']].to_csv('primeiro.csv',index=False)
