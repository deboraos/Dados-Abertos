#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import module for data manipulation
import pandas as pd

# Import module for linear algebra
import numpy as np

df = pd.read_csv('D:/Meus Downloads/TCC/bases/Mortalidade_Geral_2016_2020.csv', low_memory=False)


# In[2]:


df.columns


# In[3]:


df = df.drop(columns=['ACIDTRAB', 'ALTCAUSA', 'ASSISTMED', 'ATESTADO', 'ATESTANTE',
                      'CAUSABAS_O', 'CAUSAMAT', 'CB_PRE', 'CIRCOBITO', 'CIRURGIA',
                      'CODESTAB', 'CODIFICADO', 'CODMUNNATU', 'CODMUNRES',
                      'COMUNSVOIM', 'CONTADOR', 'DIFDATA', 'DTATESTADO', 'DTCADASTRO',
                      'DTCADINF', 'DTCADINV', 'DTCONCASO', 'DTCONINV', 'DTINVESTIG', 
                      'DTRECEBIM', 'DTRECORIGA', 'ESC', 'ESC2010', 'ESCFALAGR1'])


# In[4]:


df = df.drop(columns=['ESCMAE', 'ESCMAE2010', 'ESCMAEAGR1', 'ESTABDESCR', 'ESTCIV', 'EXAME',
                      'FONTE', 'FONTEINV', 'FONTES', 'FONTESINF', 'GESTACAO', 'GRAVIDEZ',
                      'HORAOBITO', 'IDADE','IDADEMAE', 'LINHAA', 'LINHAB', 'LINHAC',
                      'LINHAD', 'LINHAII', 'LOCOCOR', 'MORTEPARTO', 'NATURAL', 'NECROPSIA'])


# In[5]:


df = df.drop(columns=['NUDIASINF', 'NUDIASOBCO', 'NUDIASOBIN', 'NUMEROLOTE', 'OBITOGRAV',
                      'OBITOPARTO', 'OBITOPUERP', 'OCUP', 'OCUPMAE', 'ORIGEM', 'PARTO',
                      'PESO', 'QTDFILMORT', 'QTDFILVIVO', 'SEMAGESTAC',
                      'SERIESCFAL', 'SERIESCMAE', 'STCODIFICA', 'STDOEPIDEM',
                      'STDONOVA', 'TIPOBITO', 'TPMORTEOCO', 'TPNIVELINV', 'TPOBITOCOR',
                      'TPPOS', 'TPRESGINFO', 'VERSAOSCB', 'VERSAOSIST'])


# In[6]:


df.head()


# In[7]:


df.dtypes


# In[8]:


# converting the int to string format 
df.DTOBITO = df.DTOBITO.astype(str)
df.CODMUNOCOR = df.CODMUNOCOR.astype(str)

df.dtypes


# In[9]:


df.head()


# In[10]:


# converting the float to datetime format 
df['DTOBITO'] = pd.to_datetime(df['DTOBITO'], format='%d%m%Y', errors='coerce')
df['DTNASC'] = pd.to_datetime(df['DTNASC'], format='%d%m%Y', errors='coerce')

df.dtypes


# In[11]:


df.head()


# In[12]:


df['IDADE'] = df.apply(lambda row: row.DTOBITO.year - row.DTNASC.year, axis = 1)

df.head()


# In[13]:


df = df.drop(columns=['DTNASC', 'DTOBITO'])


# In[14]:


df.describe().transpose()


# In[15]:


round(df.isnull().mean() * 100,2)


# In[16]:


print('Total de idivíduos no dataset {}'.format(df.shape[0]))
print('Total de indivídios com todos os registros preenchidos (linhas) {}'.format(df.dropna().shape[0]))
print('Percentual de dados com 100% do preenchimento dos dados {}'.format(round(df.dropna().shape[0] / len(df)*100,2)))


# In[17]:


df.dropna(subset=['IDADE'],inplace=True)
df.dropna(subset=['RACACOR'],inplace=True)


# In[18]:


print('Percentual de dados com 100% do preenchimento dos dados {}'.format(round(df.dropna().shape[0] / len(df)*100,2)))


# In[19]:


cat_col = [var for var in df.columns if df[var].dtype == 'O']
df.loc[:,cat_col].head() 


# In[20]:


num_col = [var for var in df.columns if df[var].dtype != 'O']
df.loc[:,num_col].head()


# In[21]:


df['SEXO'] = df['SEXO'].replace([1, 2, 9, 0], ['Masculino', 'Feminino', 'Ignorado', 'Ignorado'])
df['RACACOR'] = df['RACACOR'].replace([1, 2, 3, 4, 5, 9], ['Branca', 'Preta', 'Amarela', 'Parda', 'Indigena' , 'Ignorado'])
df


# In[22]:


grupo_idade = []

for i in df['IDADE']:
    if i < 13:
        grupo_idade.append('Crianca')
    elif i >= 13 and i < 19:
        grupo_idade.append('Adolescente')
    elif i >= 19 and i < 60:
        grupo_idade.append('Adulto')
    else:
        grupo_idade.append('Idoso')

df['GRUPOIDADE'] = grupo_idade
df


# In[23]:


df = df.drop(columns=['IDADE'])
df


# In[24]:


df.info()


# In[25]:


df.select_dtypes('object').nunique()


# In[26]:


# Select the categorical columns
cols = df.select_dtypes('object').columns
df_cat = df[cols]

df_cat.head()


# In[27]:


# Check missing value
df_cat.isna().sum()


# In[28]:


# Convert dataframe to matrix
dfMatrix = df_cat.loc[:].to_numpy()
dfMatrix


# In[29]:


# Import module for k-protoype cluster
from kmodes.kmodes import KModes
import time

inicio = time.time()

# Choosing optimal K
cost = []
for cluster in range(1, 8):
    kmodes = KModes(n_jobs = -1, n_clusters = cluster, init = 'Huang',  n_init=2, verbose=1)
    kmodes.fit_predict(dfMatrix)
    cost.append(kmodes.cost_)
    print('Cluster initiation: {}'.format(cluster))

fim = time.time()
print('Tempo de execução em segundos: ', fim - inicio)


# In[66]:


# Converting the results into a dataframe and plotting them
df_cost = pd.DataFrame({'Cluster': range(1, 8), 'Cost': cost})


# In[67]:


# Import module for data visualization
from plotnine import *
import plotnine
# Data visualization with matplotlib
import matplotlib.pyplot as plt
# Use the theme of ggplot
plt.style.use('ggplot')

# Data viz
plotnine.options.figure_size = (8, 4.8)
(
    ggplot(data = df_cost)+
    geom_line(aes(x = 'Cluster',
                  y = 'Cost'))+
    geom_point(aes(x = 'Cluster',
                   y = 'Cost'))+
    geom_label(aes(x = 'Cluster',
                   y = 'Cost',
                   label = 'Cluster'),
               size = 8,
               nudge_y = 1000) +
    labs(title = 'Optimal number of cluster with Elbow Method')+
    xlab('Number of Clusters k')+
    ylab('Cost')+
    theme_minimal()
)


# In[68]:


# Fit the cluster

inicio = time.time()

kmodes = KModes(n_jobs = -1, n_clusters = 5, init = 'Huang',  n_init=2, verbose=1)

fim = time.time()

kmodes.fit_predict(dfMatrix)

print('Tempo de execução em segundos: ',fim - inicio)


# In[69]:


# Cluster centorid
kmodes.cluster_centroids_

# Check the iteration of the clusters created
kmodes.n_iter_

# Check the cost of the clusters created
kmodes.cost_


# In[70]:


# Add the cluster to the dataframe
df_cat['Cluster Labels'] = kmodes.labels_
df_cat['Segment'] = df_cat['Cluster Labels'].map({0:'First', 1:'Second', 2:'Third', 3:'Fourth', 4:'Fifth'})


# In[71]:


# Order the cluster
df_cat['Segment'] = df_cat['Segment'].astype('category')
df_cat['Segment'] = df_cat['Segment'].cat.reorder_categories(['First', 'Second', 'Third', 'Fourth','Fifth'])


# In[72]:


# Columns for centroids
list_col = ['customerID', 'Cluster Labels', 'Segment']
cols = [col for col in df_cat if col not in list_col]


# In[73]:


# Create an index for cluster interpretation
index = ['First Cluster', 'Second Cluster', 'Third Cluster', 'Fourth Cluster','Fifth Cluster']


# In[74]:


# Create the data frame
pd.DataFrame(kmodes.cluster_centroids_, columns = cols, index = index)


# In[75]:


df


# In[76]:


df_copy = df.copy()


# In[77]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)
df.head()


# In[78]:


fitClusters_huang = kmodes.fit_predict(df)


# In[79]:


# Predicted Clusters
fitClusters_huang


# In[80]:


clusterCentroidsDf = pd.DataFrame(kmodes.cluster_centroids_)
clusterCentroidsDf.columns = df.columns


# In[81]:


# Mode of the clusters
clusterCentroidsDf


# In[82]:


df = df_copy.reset_index()


# In[84]:


clustersDf = pd.DataFrame(fitClusters_huang)
clustersDf.columns = ['cluster_predicted']
combinedDf = pd.concat([df, clustersDf], axis = 1).reset_index(drop=True)
combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)


# In[85]:


combinedDf.head()


# In[86]:


cluster_0 = combinedDf[combinedDf['cluster_predicted'] == 0]
cluster_1 = combinedDf[combinedDf['cluster_predicted'] == 1]
cluster_2 = combinedDf[combinedDf['cluster_predicted'] == 2]
cluster_3 = combinedDf[combinedDf['cluster_predicted'] == 3]
cluster_4 = combinedDf[combinedDf['cluster_predicted'] == 4]


# In[87]:


cluster_0.info()


# In[88]:


cluster_1.info()


# In[89]:


cluster_2.info()


# In[90]:


cluster_3.info()


# In[91]:


cluster_4.info()


# In[92]:


combinedDf


# In[93]:


combinedDf.to_csv("cluster.csv")


# In[ ]:




