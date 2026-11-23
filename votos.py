import pandas as pd
from zipfile import ZipFile
for ano in [2018,2022]:
    with ZipFile(f'tse{ano}.zip') as z:
        nome=next(n for n in z.namelist() if n.endswith('_BR.csv'))
        dados=pd.read_csv(z.open(nome),encoding='latin1',sep=';',usecols=['SG_UF','CD_MUNICIPIO','NM_MUNICIPIO','NR_TURNO','NR_CANDIDATO','QT_VOTOS_NOMINAIS','DS_CARGO'])
    dados=dados[(dados.NR_TURNO==2)&(dados.SG_UF!='ZZ')&(dados.DS_CARGO.str.upper()=='PRESIDENTE')]
    votos=dados.groupby(['SG_UF','CD_MUNICIPIO','NM_MUNICIPIO','NR_CANDIDATO']).QT_VOTOS_NOMINAIS.sum().unstack()
    assert len(votos)==5570
    votos.to_csv(f'v{ano}.csv')
