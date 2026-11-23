import re,unicodedata,json,difflib,zipfile
from pathlib import Path
import pandas as pd,numpy as np
def norm(s): return re.sub('[^A-Z0-9]','',unicodedata.normalize('NFKD',s.upper()).encode('ascii','ignore').decode())
pop=pd.DataFrame(json.load(open('pop.json'))[1:]); names={}
for r in pop.to_dict('records'):
 name,uf=r['D1N'].rsplit(' - ',1);names[(uf,norm(name))]=int(r['D1C'])
rec=pd.read_csv('receita.csv',sep=';',encoding='latin1')
codigos_validos = set(names.values())
for _,id,tom,ibge,uf in rec.itertuples(index=False,name=None):
 if str(id).isdigit() and int(id) in codigos_validos:
  for n in [tom,ibge]:
   if isinstance(n,str):names.setdefault((uf,norm(n)),int(id))
fol=[]
for line in Path('folha.txt').read_text().splitlines():
 m=re.match(r'^\s*([A-Z]{2})\s+(.+?)\s+([0-9.]+)\s+R\$',line)
 if m:fol.append((m[1],m[2].strip(),int(m[3].replace('.',''))))

exceptions=[2306306,2401305,2410306,2513653,2513968,2515401,2516409,3122900,3165206,3305901,3515004,3516101,3530805,3550001,4212809,4213906,5107008]
expected=['ITAPAGE','AUGUSTO SEVERO','PRESIDENTE JUSCELINO','SANTAREM','SAO DOMINGOS DE POMBAL','SERIDO','CAMPO DE SANTANA','DONA EUSEBIA','SAO THOME DAS LETRAS','TRAJANO DE MORAIS','EMBU','FLORINIA','MOJI MIRIM','SAO LUIS DO PARAITINGA','PICARRAS','PRESIDENTE CASTELO BRANCO','POXOREO']
aliases=dict(zip(expected,exceptions))
assert set(expected) == {n for u,n,v in fol if (u,norm(n)) not in names}
d=pd.DataFrame([(names.get((u,norm(n)),aliases.get(n)),u,n,v) for u,n,v in fol],columns=['codigo_ibge','uf','municipio_folha','familias']).set_index('codigo_ibge')
assert d.index.is_unique and not d.index.hasnans and len(d)==5570 and d.familias.sum()==20653849
def sidra(k):
 a=pd.DataFrame(json.load(open(k+'.json'))[1:]);return pd.Series(pd.to_numeric(a.V.replace('-', '0')).values,index=a.D1C.astype(int))
for k in ['pop','edu','renda','dom','urb']: d[k]=sidra(k)
d['urb']*=100/d['pop']
for field,seq in [('jovens',range(6)),('idosos',range(6,14))]:
 d[field]=sum(sidra('age'+str(i)) for i in seq)*100/d['pop']
reg={v['id']:v['regiao-imediata']['id'] for v in json.load(open('municipios.json')) if v['regiao-imediata']}
d['rgi']=d.index.map(reg)
z=zipfile.ZipFile('tsecross.zip')
cross=pd.read_csv(z.open('municipio_tse_ibge.csv'),sep=';',encoding='latin1').drop_duplicates(['SG_UF','CD_MUNICIPIO_TSE'])
for year,other in [(2018,'17'),(2022,'22')]:
 v=pd.read_csv(f'v{year}.csv').merge(cross,left_on=['SG_UF','CD_MUNICIPIO'],right_on=['SG_UF','CD_MUNICIPIO_TSE'],validate='one_to_one').set_index('CD_MUNICIPIO_IBGE')
 assert len(v)==5570
 d[f'validos{year}']=v['13']+v[other]; d[f'pt{year}']=v['13']/d[f'validos{year}']
s=pd.read_csv('siafi.csv',sep=';',encoding='latin1',header=None,usecols=[0,4]); n=pd.read_csv('nis_ag.csv')
d['nis']=n.merge(s,left_on='siafi',right_on=0,validate='one_to_one').set_index(4).nis
assert d.notna().all().all() and d.nis.sum()==20065780
d['folha']=1000*d.familias/d['pop'];d['fav']=1000*d.nis/d['pop']
d.to_csv('base_municipal.csv')
print(d[['folha','fav','pop','renda','edu','jovens','idosos','urb']].describe().round(3))
