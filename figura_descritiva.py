"""Gera os painéis descritivos da Figura 1 usando os quatro agregados eleitorais e administrativos da base."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

pasta = Path(__file__).resolve().parent
base = pd.read_csv(pasta / 'base_municipal.csv')
sedes = pd.read_csv(pasta / 'coords.csv')[['codigo_ibge', 'latitude', 'longitude']]
dados = base.merge(sedes, on='codigo_ibge', how='left', validate='one_to_one')
assert len(dados) == 5570 and dados[['latitude','longitude','folha','pt2022']].notna().all().all()
dados['votos'] = 100 * dados.pt2022
regioes = {'Norte':'AC AM AP PA RO RR TO','Nordeste':'AL BA CE MA PB PE PI RN SE','Centro-Oeste':'DF GO MS MT','Sudeste':'ES MG RJ SP','Sul':'PR RS SC'}
plt.rcParams.update({'font.size':8.4,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
fig = plt.figure(figsize=(7.4,4.42))
grade = fig.add_gridspec(2,2,height_ratios=[0.8,1.45],hspace=.72,wspace=.38,left=.13,right=.98,bottom=.11,top=.94)
for j,(campo,rotulo) in enumerate([('folha','Famílias na folha por mil habitantes'),('votos','Votos válidos em Lula (%)')]):
 ax=fig.add_subplot(grade[0,j])
 grupos=[dados.loc[dados.uf.isin(ufs.split()),campo] for ufs in regioes.values()]
 ax.boxplot(grupos,vert=False,showfliers=False,widths=.54,patch_artist=True,boxprops={'facecolor':'#bdd6df','edgecolor':'#3c5864'},medianprops={'color':'#172e40'},whiskerprops={'color':'#596e77'},capprops={'color':'#596e77'})
 ax.set_yticks(range(1,6),regioes.keys()); ax.invert_yaxis();ax.set_xlabel(rotulo);ax.tick_params(length=2,labelsize=7.9)
 ax.set_title('A. Famílias por região' if j==0 else 'B. Votação por região',loc='left',fontsize=9.6,weight='bold')
ax=fig.add_subplot(grade[1,0]);ax.scatter(dados.folha,dados.votos,s=4,color='#266779',alpha=.23,linewidths=0,rasterized=True)
ax.set(xlabel='Famílias na folha por mil habitantes',ylabel='Votos válidos em Lula (%)',xlim=(0,485),ylim=(0,100))
ax.set_title('C. Associação municipal',loc='left',fontsize=9.6,weight='bold')
ax=fig.add_subplot(grade[1,1])
cores=['#c6dce5','#8ebcc6','#407f87','#163d54']
quartis=pd.qcut(dados.folha,4,labels=False)
for grupo,cor in enumerate(cores):
 sub=dados[quartis==grupo]; ax.scatter(sub.longitude,sub.latitude,s=2.7,color=cor,linewidths=0,rasterized=True)
ax.set_aspect(1/np.cos(np.deg2rad(15)));ax.set_axis_off()
ax.set_title('D. Famílias por mil nas sedes municipais',fontsize=9.2,weight='bold')
fig.legend([Line2D([],[],marker='o',ls='',ms=5,color=c) for c in cores],['1º','2º','3º','4º'],ncol=4,title='Quartil de famílias por mil',loc='lower center',bbox_to_anchor=(.79,-.012),frameon=False,columnspacing=.6,handletextpad=.2,fontsize=7.5,title_fontsize=7.5)
fig.savefig(pasta/'figura_descritiva.pdf',dpi=240)
plt.close(fig)
