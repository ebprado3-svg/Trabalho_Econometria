"""Sensibilidades e estatística espacial da sequência complementar."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.stats import t,norm
import matplotlib.pyplot as plt

def analisar(d,base,design,fit,gr,y,h,x,xyz,i,k):
    n=len(y);rows=[];vifs=[];mor=[];con=[]
    names=['F','H','ln_renda','edu','urb','jovens','idosos','ln_pop']
    termos=pd.DataFrame(dict(F=x,H=h,ln_renda=np.log(d.renda),edu=d.edu,urb=d.urb,jovens=d.jovens,idosos=d.idosos,ln_pop=np.log(d['pop'])))
    for j,q in [(3,2),(4,4),(5,8)]:
        X=termos.iloc[:,:q]
        for name,T in [('dentro de UF',X-X.groupby(d.uf).transform('mean')),('com dummies',pd.concat([X,pd.get_dummies(d.uf,drop_first=True,dtype=float)],axis=1))]:
            vs=np.diag(np.linalg.inv(T.corr().to_numpy()))
            vifs.extend((j,name,names[m],vs[m]) for m in range(q))
    pd.DataFrame(vifs,columns=['M','definicao','termo','VIF']).to_csv('vif.csv',index=False)
    def recorte(mask,j,nome,w=None):
        z=d.loc[mask]
        X=np.column_stack([np.ones(len(z)),x[mask],pd.get_dummies(z.uf,drop_first=True,dtype=float),h[mask]])
        if j>=4:X=np.column_stack([X,np.log(z.renda),z.edu])
        b,_,_,se,_,_=fit(X,y[mask],None if w is None else np.asarray(w)[mask],gr[mask])
        q=t.ppf(.975,z.uf.nunique()-1)
        rows.append([nome,j,len(z),z.uf.nunique(),b[1],b[1]-q*se[1],b[1]+q*se[1]])
    mask=d.familias<=d.dom
    assert (~mask).sum()==110
    recorte(mask,3,'sem excesso de famílias')
    grandes=d.nlargest(int(np.ceil(n*.01)),['pop','codigo_ibge'])
    for j in [3,4]:
        for peso,w in [('igual',None),('população',d['pop']),('votos',d.validos2022)]:
            recorte(~d.index.isin(grandes.index),j,'sem 1% maior, '+peso,w)
    for j,mask in enumerate(pd.qcut(h,3,labels=False)==a for a in range(3)):
        recorte(mask,3,f'tercil {j+1} de Haddad')
    pd.DataFrame(rows,columns=['teste','M','n','G','b','lo','hi']).to_csv('sensibilidades.csv',index=False)
    # Kernel triangular de Conley. Todas as distâncias são de grande círculo.
    R=6371.0088
    a,b=cKDTree(xyz).query_pairs(2*np.sin(200/(2*R)),output_type='ndarray').T
    dist=2*R*np.arcsin(np.linalg.norm(xyz[a]-xyz[b],axis=1)/2)
    def kernel(km):
        m=dist<km;v=1-dist[m]/km;ix=np.arange(n)
        return csr_matrix((np.r_[v,v,np.ones(n)],(np.r_[a[m],b[m],ix],np.r_[b[m],a[m],ix])),shape=(n,n))
    kernels={km:kernel(km) for km in [50,100,200]}
    nets={'oito vizinhos':np.ones(len(i),bool),'entre RGI':d.rgi.values[i]!=d.rgi.values[k],'entre UF':gr[i]!=gr[k]}
    for j in [2,3,4,5]:
        X=design(j);beta,e,A,*_=fit(X);z=e-e.mean()
        for label,mascara in nets.items():
            u,v=i[mascara],k[mascara];fator=n/len(u)/(z@z)
            def indice(r):return fator*(r[u]@r[v])
            obs=indice(z);E=-1/(n-1)
            rng=np.random.default_rng(202211)
            count=sum(abs(indice(rng.permutation(z))-E)>=abs(obs-E) for _ in range(999))
            mor.append([j,label,len(u),obs,(1+count)/1000])
        psi=(X@A[:,1])*e
        for km,W in kernels.items():
            se=np.sqrt(psi@(W@psi));q=norm.ppf(.975)
            con.append([j,km,beta[1],se,beta[1]-q*se,beta[1]+q*se])
    pd.DataFrame(mor,columns=['M','rede','arestas','I','p_perm']).to_csv('moran.csv',index=False)
    pd.DataFrame(con,columns=['M','km','b','ep','lo','hi']).to_csv('conley.csv',index=False)
    U=pd.get_dummies(d.uf,dtype=float);G=U.shape[1]
    X=np.column_stack([U,U.to_numpy()*x[:,None],h])
    beta,e,A,*_=fit(X,g=d.rgi.to_numpy())
    psi=(X@A[:,G:2*G])*e[:,None]
    se=np.sqrt(np.sum(psi*(kernels[100]@psi),axis=0))
    b=beta[G:2*G];delta=norm.ppf(.975)*se
    f=pd.DataFrame(dict(uf=U.columns,b=b,ep=se,lo=b-delta,hi=b+delta)).sort_values('b')
    f.to_csv('inclinacoes_uf.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.7,4.95),layout='constrained')
    ax.errorbar(f.b,np.arange(G),xerr=[f.b-f.lo,f.hi-f.b],fmt='o',ms=3,color='#21536e',ecolor='#5e7d8d',lw=1.1,capsize=2)
    ax.axvline(0,color='#444',lw=.8);ax.set_yticks(np.arange(G),f.uf,fontsize=8)
    ax.grid(axis='x',lw=.55,color='#ddd');ax.set_xlabel('Pontos percentuais na votação em Lula por dez famílias por mil')
    ax.spines[['top','right']].set_visible(False);fig.savefig('floresta.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,2.25),layout='constrained')
    for j,grupo in pd.read_csv('decis.csv').groupby('M'):
        if j in (2,3,5):ax.plot(grupo.decil,grupo.residuo_pp,marker='o',ms=3,label=f'M{j}',lw=1.4)
    ax.axhline(0,color='#444',lw=.7);ax.set(xticks=range(1,11),xlabel='Decil de famílias na folha por mil',ylabel='Resíduo médio (p.p.)')
    ax.legend(frameon=False,ncols=3);ax.spines[['top','right']].set_visible(False);fig.savefig('decis.pdf');plt.close(fig)
