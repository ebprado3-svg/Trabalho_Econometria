"""Inferência nas equações originais, diagnósticos e efeito marginal da média fracionária."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import t,chi2

def calcular(base):
    d=base.merge(pd.read_csv(Path(__file__).with_name('favorecidos_out2021.csv')),on='codigo_ibge',validate='one_to_one')
    assert len(d)==5570
    y=100*d.pt2022.to_numpy()
    d['H']=100*d.pt2018;d['Y']=y;d['dif']=d.Y-d.H;d['saldo']=1000*(d.nis-d.nis21)/d['pop']
    d[['Y','folha','edu','renda','H','dif','saldo']].describe().T.to_csv('descritivas.csv',index_label='variavel')
    rgi=pd.factorize(d.rgi)[0];tam=np.bincount(rgi)
    def moran(z):
        soma=np.bincount(rgi,weights=z)
        return len(z)/(tam[rgi]>1).sum()*(z@((soma[rgi]-z)/np.maximum(tam[rgi]-1,1)))/(z@z)
    rows=[];dg=[]
    for j,adicionais in enumerate([[],['edu'],['edu','ln_renda']],1):
        X=np.column_stack([np.ones(len(d)),d.folha/10,*[d[q] for q in adicionais]])
        b=np.linalg.lstsq(X,y,rcond=None)[0];e=y-X@b;A=np.linalg.inv(X.T@X);N,K=X.shape
        for nome,g in [('UF',d.uf),('RGI',d.rgi),('HC1',d.codigo_ibge)]:
            scores=pd.DataFrame(X*e[:,None]).groupby(g.to_numpy()).sum().to_numpy()
            G=len(scores);se=np.linalg.norm(scores@A,axis=0)*np.sqrt(G/(G-1)*(N-1)/(N-K))
            q=t.ppf(.975,(N-K) if nome=='HC1' else G-1)
            for pos,termo in enumerate(['intercepto','F',*adicionais]):
                escala=np.log(1.1) if termo=='ln_renda' else 1.
                rows.append([j,termo,nome,b[pos]*escala,se[pos]*escala,(b[pos]-q*se[pos])*escala,(b[pos]+q*se[pos])*escala,N,1-e@e/((y-y.mean())@(y-y.mean()))])
        z=e-e.mean();I=moran(z);rng=np.random.default_rng(202211);E=-1/(N-1)
        p=(1+sum(abs(moran(rng.permutation(z))-E)>=abs(I-E) for _ in range(999)))/1000
        ee=e**2;pred=X@np.linalg.lstsq(X,ee,rcond=None)[0];lm=N*(1-sum((ee-pred)**2)/sum((ee-ee.mean())**2))
        lev=np.sum((X@A)*X,axis=1);cook=ee*lev/(K*(ee.sum()/(N-K))*(1-lev)**2)
        medias=pd.Series(e).groupby(pd.qcut(d.folha,10,labels=False)).mean()
        dg.append([j,I,p,lm,chi2.sf(lm,K-1),sum(y-e>100),sum(abs(medias)>5),abs(medias).max(),sum(cook>4/N),np.linalg.inv(pd.DataFrame(X[:,1:]).corr()).diagonal().max()])
    pd.DataFrame(rows,columns=['eq','termo','erro','b','ep','lo','hi','n','r2']).to_csv('referencia.csv',index=False)
    pd.DataFrame(dg,columns=['eq','moran','p_perm','LM','p_koenker','acima_100','decis_acima5','max_decil','cook','vif']).to_csv('diagnosticos_referencia.csv',index=False)
    z=(d.folha.to_numpy()-d.folha.mean())/10;X=np.column_stack([np.ones(len(z)),z])
    f=minimize(lambda a:np.sum(np.logaddexp(0,X@a)-(y/100)*(X@a)),np.zeros(2),jac=lambda a:X.T@(expit(X@a)-y/100),method='BFGS')
    assert abs(f.jac).max()<1e-4
    mu=expit(X@f.x);w=mu*(1-mu);A=np.linalg.inv(X.T@(X*w[:,None]))
    S=pd.DataFrame(X*(y/100-mu)[:,None]).groupby(d.uf.to_numpy()).sum().to_numpy()@A
    G=len(S);n=len(y);V=S.T@S*G/(G-1)*(n-1)/(n-2)
    grad=100*(np.array([0.,w.mean()])+f.x[1]*(w*(1-2*mu))@X/n)
    rows=[]
    for nome,b,g in [('intercepto',f.x[0],[1,0]),('z',f.x[1],[0,1]),('efeito marginal médio',100*f.x[1]*w.mean(),grad)]:
        ep=np.sqrt(np.asarray(g)@V@g);q=t.ppf(.975,G-1);rows.append([nome,b,ep,b-q*ep,b+q*ep])
    pd.DataFrame(rows,columns=['termo','b','ep','lo','hi']).to_csv('logit.csv',index=False)
    ufs={'Norte':'AC AM AP PA RO RR TO','Nordeste':'AL BA CE MA PB PE PI RN SE','Centro-Oeste':'DF GO MS MT','Sudeste':'ES MG RJ SP','Sul':'PR RS SC'}
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained')
    for ax,c,titulo in zip(axes,['folha','Y'],['Famílias na folha por mil habitantes','Votos válidos em Lula (%)']):
        ax.boxplot([d.loc[d.uf.isin(s.split()),c] for s in ufs.values()],vert=False,showfliers=False,patch_artist=True,boxprops={'facecolor':'#d3e4ea'},medianprops={'color':'#184c63'})
        ax.set_yticks(range(1,6),list(ufs));ax.invert_yaxis();ax.set_xlabel(titulo);ax.spines[['top','right']].set_visible(False)
    fig.savefig('regioes.pdf');plt.close(fig)
