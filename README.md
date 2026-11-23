# Auxílio Brasil e voto em Lula em 2022

Dados municipais e código de reprodução do artigo de Bruno Donadel Mainieri e Enrico Bertolini Prado. A unidade de observação é o município. A base principal tem 5.570 linhas e a amostra complementar tem 5.569, com exclusão de Brasília. O artigo está em [Artigo_Econometria_corrigido.pdf](Artigo_Econometria_corrigido.pdf).

## Reprodução da análise

Com Python 3.12, extraia ou clone todos os arquivos na mesma pasta e, a partir dela, execute:

```bash
python -m pip install -r requirements.txt
python reestimar.py
python modelos_centrais.py
python expansao.py
python coeficientes_m5.py
python figura_descritiva.py
```

O segundo comando de análise inclui o bootstrap por UF e pode levar mais tempo. O arquivo [LEIA-ME.txt](LEIA-ME.txt) documenta a semente, a correspondência entre tabelas e saídas, a reconstrução dos insumos brutos e o uso de IA por etapa.

| Arquivo | Conteúdo |
| --- | --- |
| base_municipal.csv | Votação, folha programada, NIS agregados e controles censitários, por código IBGE |
| primeiro.csv | Votação de Haddad no primeiro turno de 2018 |
| favorecidos_out2021.csv | NIS distintos no município em outubro de 2021 |
| coords.csv | Coordenadas das sedes, identificadas por código IBGE |
| fontes.json | Endereços dos insumos originais |
| acessos_referencias.csv | Data de consulta das referências completadas nesta revisão |
| figura_descritiva.pdf | Boxplots regionais, dispersão municipal e mapa das sedes |
| coeficientes_m5.csv | Contrastes das covariáveis em pontos percentuais de votação e intervalos por UF |

Os arquivos de análise utilizam agregados municipais. Para recuperar a base principal a partir das fontes originais, siga a seção 3 do LEIA-ME.
