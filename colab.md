Link do cola: https://colab.research.google.com/drive/14AfpHGEAajnBu4fnKcpudriJYbRmZwus

Código utilizado.

---

import pandas as pd

df_dados_criminais_2021 = pd.read_csv('dados_criminais/dados_criminais_2021.csv', delimiter=';' ,encoding="iso-8859-1")
print(df_dados_criminais_2021.head())

df_dados_criminais_2021 = pd.read_csv('dados_criminais/dados_criminais_2021.csv', delimiter=';' ,encoding="iso-8859-1")
print(df_dados_criminais_2021.head())

# Padroniza os nomes das colunas em uma única linha

df*criminalidade.columns = df_criminalidade.columns.str.strip().str.lower().str.replace(' ', '*', regex=True)

# Identifica e remove as colunas que não têm nome ('unnamed')

colunas_vazias = [col for col in df_criminalidade.columns if 'unnamed' in col]
df_criminalidade = df_criminalidade.drop(columns=['...'])
df_criminalidade = df_criminalidade.drop(columns=colunas_vazias)

mapeamento_renomear = {
'quantidade_vítimas': 'quantidade_vitimas',
'idade_vítima': 'idade_vitima',
'sexo_vítima': 'sexo_vitima',
'cor_vítima': 'cor_vitima',
'sequência': 'sequencia',
}

# Renomeia as colunas usando o dicionário

df_criminalidade = df_criminalidade.rename(columns=mapeamento_renomear)

# Converte a coluna 'quantidade_vitimas' para int

df_criminalidade['quantidade_vitimas'] = (
df_criminalidade['quantidade_vitimas']
.fillna(0) # preenche NaN com 0
.astype(str) # transforma em string para limpeza
.str.replace(r'[^0-9-]', '', regex=True) # remove tudo que não é número ou sinal de negativo
.replace('', 0) # se ficou vazio, vira 0
.astype(int) # converte para inteiro
)

# Converte a coluna 'idade_vitima' para int

df_criminalidade['idade_vitima'] = (
df_criminalidade['idade_vitima']
.fillna(0)
.astype(str)
.str.replace(r'[^0-9-]', '', regex=True)
.replace('', 0)
.astype(int)
)

# Converte a coluna 'data_fato' para datetime

df_criminalidade['data_fato'] = pd.to_datetime(
df_criminalidade['data_fato'],
format='%d/%m/%Y',
errors='coerce'
)

print(df_criminalidade.info())

# Preenchendo NaNs em colunas categóricas com "Nao Informado"

colunas_categoricas = ['grupo_fato', 'tipo_enquadramento', 'tipo_fato', 'municipio_fato', 'local_fato', 'bairro', 'sexo_vitima', 'cor_vitima', 'hora_fato', 'sequencia']
for col in colunas_categoricas:
if col in df_criminalidade.columns:
df_criminalidade[col] = df_criminalidade[col].fillna('Nao Informado')

# Remover duplicatas

linhas_duplicadas = df_criminalidade.duplicated().sum()
df_criminalidade.drop_duplicates(inplace=True)

df_meteorologia = pd.read_csv(
'dados_metereologicos/dados_metereologicos.csv',
delimiter=';',
encoding='iso-8859-1',
skiprows=9, # Pula as 9 primeiras linhas de metadados
decimal=',' # Usa a vírgula como separador decimal
)
df_meteorologia

# padroniza colunas

df_meteorologia = df_meteorologia.rename(columns={
'Data Medicao': 'data_fato',
'PRECIPITACAO TOTAL, DIARIO (AUT)(mm)': 'precipitacao_total_mm',
'TEMPERATURA MAXIMA, DIARIA (AUT)(Â°C)': 'temperatura_maxima_c',
'TEMPERATURA MINIMA, DIARIA (AUT)(Â°C)': 'temperatura_minima_c',
'UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)': 'umidade_media_pct',
'VENTO, VELOCIDADE MEDIA DIARIA (AUT)(m/s)': 'vento_medio_ms'
})

# remove coluna inutil

df_meteorologia = df_meteorologia.drop(columns=['Unnamed: 6'])

# padroniza datas e cria coluna com a cidade

df_meteorologia['data_fato'] = pd.to_datetime(df_meteorologia['data_fato'], format='%Y-%m-%d')

# filtra os dados de criminalidade para Passo Fundo

df_criminalidade_pf = df_criminalidade[df_criminalidade['municipio_fato'] == 'PASSO FUNDO'].copy()

# merge dos datasets

df_final = pd.merge(
df_criminalidade_pf,
df_meteorologia,
on=['data_fato'],
how='left'
)

# trata colunas vazias após o merge

colunas_para_preencher = [
'precipitacao_total_mm',
'temperatura_maxima_c',
'temperatura_minima_c',
'umidade_media_pct',
'vento_medio_ms'
]
df_final[colunas_para_preencher] = df_final[colunas_para_preencher].fillna(method='ffill')
df_final[colunas_para_preencher] = df_final[colunas_para_preencher].fillna(method='bfill')

# verificaçao se o merge deu certo

df_final

# OBS: Este dataset é uma versão ATUALIZADA (setembro/2025), então os números de outliers

# podem diferir da análise feita anteriormente em outra máquina ou versão do arquivo.

# A detecção de outliers é feita pelo método IQR, que depende dos quartis Q1 e Q3,

# e portanto, varia se o dataset for maior ou tiver valores alterados.

import numpy as np

# colunas numéricas para verificar outliers

colunas_numericas = [
'quantidade_vitimas',
'idade_vitima',
'precipitacao_total_mm',
'temperatura_maxima_c',
'temperatura_minima_c',
'umidade_media_pct',
'vento_medio_ms'
]

# detectar outliers usando o método IQR

def encontrar_outliers_iqr(df, coluna):
Q1 = df[coluna].quantile(0.25)
Q3 = df[coluna].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 _ IQR
limite_superior = Q3 + 1.5 _ IQR
outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]
return outliers

# encontrar e contar outliers para cada coluna

print("--- Checkpoint: Detecção de Outliers ---")
for col in colunas_numericas:
outliers = encontrar_outliers_iqr(df_final, col)
print(f"Colunha '{col}': {len(outliers)} outliers encontrados.")

from sklearn.preprocessing import StandardScaler

# colunas numéricas para padronização

colunas_para_padronizar = [
'quantidade_vitimas',
'idade_vitima',
'precipitacao_total_mm',
'temperatura_maxima_c',
'temperatura_minima_c',
'umidade_media_pct',
'vento_medio_ms'
]

# cópia para não alterar o DataFrame original

df_padronizado = df_final.copy()

# padronizador

scaler = StandardScaler()

# padronização apenas nas colunas numéricas

df_padronizado[colunas_para_padronizar] = scaler.fit_transform(df_padronizado[colunas_para_padronizar])

# Gráfico: Evolução de Crimes por Ano

# Cada ponto representa a quantidade total de crimes registrados em um ano específico.

import matplotlib.pyplot as plt

crimes_ano = df_final.groupby(df_final['data_fato'].dt.year).size()
crimes_ano.plot(kind='line', marker='o', color='red', figsize=(10,5), title='Evolução de Crimes por Ano')
plt.xlabel('Ano')
plt.ylabel('Quantidade de Crimes')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Gráfico: Top 10 Bairros com Mais Crimes

# A altura das barras representa a quantidade de crimes em cada bairro.

top_bairros = df_final['bairro'].value_counts().head(10)
top_bairros.plot(kind='barh', color='skyblue', figsize=(10,6), title='Top 10 Bairros com Mais Crimes')
plt.xlabel('Quantidade de Crimes')
plt.ylabel('Bairro')
plt.gca().invert_yaxis()
plt.show()

# Gráfico: Distribuição de Vítimas por Faixa Etária

# Cada barra representa uma faixa etária específica, facilitando a visualização de quais grupos são mais afetados.

faixas = pd.cut(df_final['idade_vitima'], bins=[0,12,18,30,50,100], labels=['0-12','13-18','19-30','31-50','51+'])
faixas.value_counts().sort_index().plot(kind='bar', color='orange', figsize=(8,5), title='Distribuição de Vítimas por Faixa Etária')
plt.xlabel('Faixa Etária')
plt.ylabel('Quantidade de Vítimas')
plt.show()

# Gráfico: Média de Vítimas por Faixa de Temperatura

# Cada barra representa uma faixa de temperatura, permitindo observar se há relação entre temperatura e quantidade de vítimas.

import numpy as np
df_final['faixa_temp'] = pd.cut(df_final['temperatura_maxima_c'], bins=np.arange(0,50,5))
media_vitimas = df_final.groupby('faixa_temp')['quantidade_vitimas'].mean()
media_vitimas.plot(kind='bar', color='green', figsize=(10,5), title='Média de Vítimas por Faixa de Temperatura')
plt.xlabel('Faixa de Temperatura (°C)')
plt.ylabel('Média de Vítimas')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# Gráfico: Tipos de Crimes com Maior Ocorrência

# A largura de cada barra representa a quantidade de ocorrências de cada tipo de crime.

top_tipos = df_final['tipo_enquadramento'].value_counts().head(10)

plt.figure(figsize=(10,6))
top_tipos.plot(kind='barh', color='skyblue')
plt.title('Tipos de Crimes com Maior Ocorrência', fontsize=16)
plt.xlabel('Quantidade de Crimes')
plt.ylabel('Tipo de Crime')
plt.gca().invert_yaxis() # maior no topo
plt.show()

# Gráfico: Proporção dos Principais Tipos de Crimes

# Cada fatia indica a proporção de ocorrências de cada tipo de crime em relação ao total.

proporcao = df_final['tipo_enquadramento'].value_counts()
top10 = proporcao.head(10)
top10['Outros'] = proporcao[10:].sum() # soma dos demais

plt.figure(figsize=(7,7))
plt.pie(
top10,
labels=top10.index,
autopct='%1.1f%%',
startangle=90,
colors=['skyblue','orange','green','red','purple','yellow','pink','brown','cyan','magenta','grey']
)
plt.title('Proporção dos Principais Tipos de Crimes', fontsize=16)
plt.show()

# Gráfico: Relação entre Temperatura Máxima e Quantidade de Vítimas

# Cada ponto representa um registro de crime

plt.figure(figsize=(10,6))
plt.scatter(df_final['temperatura_maxima_c'], df_final['quantidade_vitimas'], color='purple', alpha=0.6)
plt.title('Relação entre Temperatura Máxima e Quantidade de Vítimas')
plt.xlabel('Temperatura Máxima (°C)')
plt.ylabel('Quantidade de Vítimas')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

#Se hora_fato estiver como string com formato "HH:MM" ou similar
df_final['hora'] = pd.to_datetime(df_final['hora_fato'], errors='coerce').dt.hour

#Número de crimes por hora do dia
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.countplot(data=df_final, x='hora', palette='viridis')

plt.title('Número de Crimes por Hora do Dia', fontsize=16, weight='bold')
plt.xlabel('Hora do Dia', fontsize=12)
plt.ylabel('Número de Ocorrências', fontsize=12)
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Top 10 tipos de crime mais frequentes

top_tipos = df_final['tipo_fato'].value_counts().nlargest(10).index

# Filtrar o dataframe para esses tipos

df_top_tipos = df_final[df_final['tipo_fato'].isin(top_tipos)]

# Filtrar top 10 tipos de crime para facilitar visualização

top_tipos = df_final['tipo_fato'].value_counts().nlargest(10).index
df_top = df_final[df_final['tipo_fato'].isin(top_tipos)]

# Criar tabela agregada

tabela = pd.crosstab(
index=[df_top['tipo_fato'], df_top['sexo_vitima']],
columns=df_top['tipo_enquadramento']
)

#10 crimes mais frequentes por sexo da vitima
import matplotlib.pyplot as plt
import seaborn as sns

# Somar ocorrências por tipo_enquadramento e sexo_vitima

df_counts = df_final.groupby(['tipo_enquadramento', 'sexo_vitima']).size().reset_index(name='counts')

# Pegar os top 10 tipos de crime mais frequentes (somando todos os sexos)

top_crimes = df_counts.groupby('tipo_enquadramento')['counts'].sum().nlargest(10).index

# Filtrar só esses crimes

df_top = df_counts[df_counts['tipo_enquadramento'].isin(top_crimes)]

plt.figure(figsize=(14, 7))

sns.barplot(data=df_top, x='tipo_enquadramento', y='counts', hue='sexo_vitima', palette='Set2')

plt.title('Top 10 Tipos de Crime por Sexo da Vítima', fontsize=16, weight='bold')
plt.xlabel('Tipo de Crime', fontsize=14)
plt.ylabel('Número de Ocorrências', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sexo da Vítima')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
