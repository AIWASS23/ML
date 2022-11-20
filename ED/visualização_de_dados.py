import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import geopandas as gpd

# Carregando os dados
data = pd.read_csv('https://raw.githubusercontent.com/alissonSCA/dataset/master/use-of-force.csv', sep=",").set_index('ID')

# Carregando dados geograficos
precinctsGeo = gpd.read_file('https://raw.githubusercontent.com/alissonSCA/dataset/master/spd-precincts.geojson')
beatsGeo = gpd.read_file('https://raw.githubusercontent.com/alissonSCA/dataset/master/spd-beats.geojson')
## Criando o mapa dos setores a partir dos beats
#     Como não foi possível encontrar os dados geográficos dos setores
#     iremos produzi-lo a partir dos beats
#     o nome dos beats é o nome do setor concatenado com um número inteiro entre 1 e 3
sectors = [w[0] for w in beatsGeo['name']]
beatsGeo['sector'] = sectors
sectorsGeo = beatsGeo.dissolve(by='sector')
sectorsGeo['name'] = [w[0] for w in sectorsGeo['name']]

# Tratar dados faltantes
# Dados faltantes ou não presentes na descrição dos dados serão associados
# a um valor padrão '-'
## Precinct faltantes ou ausentes => '-'
data['Precinct'] = data['Precinct'].fillna('-')
data['Precinct'] = data['Precinct'].str.replace('X','-')
## Setores faltantes ou ausentes => '-'
data['Sector'] = data['Sector'].str.replace('X','-')
data['Sector'] = data['Sector'].str.replace('99','-')
## Beats faltantes ou ausentes => '-'
data['Beat'] = data['Beat'].fillna('-')
data['Beat'] = data['Beat'].str.replace('99','-')
data['Beat'] = data['Beat'].str.replace('XX','-')
## Gênero e raça ausentes => '-'
data['Subject_Race'] = data['Subject_Race'].fillna('-')
data['Subject_Gender'] = data['Subject_Gender'].fillna('-')

#Como as ocorrencias se distribuem nas delegacias e nos setores
pTotal = data['Precinct'].value_counts()
total = []
for precName in precinctsGeo['name']:
    total.append(pTotal[precName])
precinctsGeo['total'] = total

sTotal = data['Sector'].value_counts()
total = []
for secName in sectorsGeo['name']:
    total.append(sTotal[secName])
sectorsGeo['total'] = total

#Plots
fig, axes = plt.subplots(2, 1, figsize=(40, 20))
fig.subplots_adjust(hspace=0.2, wspace=0.1)

ax = precinctsGeo.plot(ax=axes.flat[0], column='total', cmap='YlGn', legend=True)
ax.set_title('Precincts')
ax.set_axis_off()
for idx, row in precinctsGeo.iterrows():
    ax.annotate(s=row['name'], xy=row['geometry'].centroid.coords[0],horizontalalignment='center')

ax = sectorsGeo.plot(ax=axes.flat[1], column='total', cmap='YlGn', legend=True)
ax.set_title('Sectors')
ax.set_axis_off()
for idx, row in sectorsGeo.iterrows():
    ax.annotate(s=row['name'], xy=row['geometry'].centroid.coords[0],horizontalalignment='center')

# Quanto um beat é "responsável" pelo resultado do seu setor?
bTotal = data['Beat'].value_counts()
total = []
relSec = []
for bName in beatsGeo['name']:
    total.append(bTotal[bName])
    relSec.append(float(bTotal[bName])/sTotal[bName[0]])
beatsGeo['total'] = total
beatsGeo['relSec'] = relSec

fig, axes = plt.subplots(figsize=(20, 10))
fig.subplots_adjust(hspace=0.2, wspace=0.1)

ax = beatsGeo.plot(ax=axes, column='relSec', cmap='YlGn', legend=True)
ax.set_title('Beat vs Sector')
ax.set_axis_off()
for idx, row in beatsGeo.iterrows():
    ax.annotate(s=row['name'], xy=row['geometry'].centroid.coords[0],horizontalalignment='center')

## Explora um setor específico

S = 'E'; # Setor a ser detalhado
v = beatsGeo[beatsGeo['sector']==S]['total']
l = beatsGeo[beatsGeo['sector']==S]['name']

fig1, ax1 = plt.subplots()
ax1.pie(v, labels = l, autopct=lambda p: '%.2f (%.0f)' % (p / 100, p * np.sum(v) / 100))
ax1.axis('equal')
plt.show()

# Em cada setor, qual a proporção entre level 2 vs total do setor?

sTotal = data['Sector'].value_counts()
level2 = []
for secName in sectorsGeo['name']:
    v = data[data['Sector']==secName]['Incident_Type'].value_counts()['Level 2 - Use of Force']    
    level2.append(float(v)/sTotal[secName])
sectorsGeo['level2'] = level2

fig, axes = plt.subplots(1, 1, figsize=(20, 10))
ax = sectorsGeo.plot(ax=axes, column='level2', cmap='YlGn', legend=True)
ax.set_title('Level 2 por setor (%)')
ax.set_axis_off()
for idx, row in sectorsGeo.iterrows():
    ax.annotate(s=row['name'], xy=row['geometry'].centroid.coords[0],horizontalalignment='center')

l1Values = []
l2Values = []
for sName in sTotal.keys():
    v = data[data['Sector']==sName]['Incident_Type'].value_counts()
    l1Values.append(v['Level 1 - Use of Force'])
    l2Values.append(v['Level 2 - Use of Force'])

fig, ax1 = plt.subplots(figsize=(12, 6))

ind = np.arange(len(l1Values))
p1 = ax1.bar(ind, l2Values)
p2 = ax1.bar(ind, l1Values, bottom=l2Values)
ax1.set_ylabel('#Ocorrencias')
ax1.set_title('Ocorrencias por setor e nivel')
plt.xticks(ind, sTotal.keys())
plt.xlabel('Setor')

ax2 = ax1.twinx()
razao = l2Values/sTotal
p3 = ax2.plot(ind, razao, 'b^-')
ax2.set_ylabel('razao')

p4 = ax2.plot(ind, np.ones(len(ind))*np.mean(razao), 'r--')

plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Level 2 - Use of Force', 'Level 1 - Use of Force', 'Level 2/Total', 'Proporcao Media'))

plt.show()

data['w_day'] = [pd.to_datetime(D, yearfirst=True).weekday() for D in data['Occured_date_time']]

l1Values = []
l2Values = []
for wd in [0,1,2,3,4,5,6]:
    v = data[data['w_day']==wd]['Incident_Type'].value_counts()
    l1Values.append(v['Level 1 - Use of Force'])
    l2Values.append(v['Level 2 - Use of Force'])
    
fig, ax1 = plt.subplots(figsize=(12, 6))

ind = np.arange(len(l1Values))
p1 = ax1.bar(ind, l2Values)
p2 = ax1.bar(ind, l1Values, bottom=l2Values)
ax1.set_ylabel('#Ocorrencias')
ax1.set_title('Ocorrencias por dia da semana')
wDayName = ['dom','seg','ter','qua','qui','sex','sab']
plt.xticks(ind, wDayName)

plt.legend((p1[0], p2[0]), ('Level 2 - Use of Force', 'Level 1 - Use of Force'))

data['hour'] = [pd.to_datetime(D, yearfirst=True).hour for D in data['Occured_date_time']]

l1Values = []
l2Values = []
for h in np.unique(data['hour']):
    v = data[data['hour']==h]['Incident_Type'].value_counts()
    l1Values.append(v['Level 1 - Use of Force'])
    l2Values.append(v['Level 2 - Use of Force'])

fig, ax1 = plt.subplots(figsize=(12, 6))

ind = np.arange(len(l1Values))
p1 = ax1.bar(ind, l2Values)
p2 = ax1.bar(ind, l1Values, bottom=l2Values)
ax1.set_ylabel('#Ocorrencias')
ax1.set_xlabel('Hora')
ax1.set_title('Ocorrencias por hora do dia')
plt.xticks(ind, np.unique(data['hour']))

plt.legend((p1[0], p2[0]), ('Level 2 - Use of Force', 'Level 1 - Use of Force'))

policiais = np.unique(data['Officer_ID'])
policiais = np.delete(policiais, np.where(policiais == 456)) # Remove outlier

total = []
razao = []
for pName in policiais:
    v = data[data['Officer_ID']==pName]['Incident_Type'].value_counts()
    if (not ('Level 2 - Use of Force' in v.keys())):        
        v['Level 2 - Use of Force'] = 0
    razao.append(float(v['Level 2 - Use of Force'])/np.sum(v))
    total.append(np.sum(v))
    if (np.sum(v) == 80):
        print(pName)

ax = sns.distplot(razao,kde=False)
ax.yaxis.set_ticklabels([])
plt.xlabel('%Level 2')

percLim = 0.6                       # Listar policiais com mais de percLim% de ocorrências level 2
t = len(policiais[np.array(razao) >= percLim]) # Alternativamente, faça t = número de policiais que deseja listar

ind = np.argsort( np.array(razao) )
print('lista dos %d policiais com proporcionalmente mais ocorrências level 2:\n'%(t))
print(policiais[ind[len(ind)-t:]])

rMean = []
rStd = []
for t in np.unique(total):
    i = np.where(total == t)[0]
    r2 = np.array(razao)[i]
    rMean.append(np.mean(r2))
    rStd.append(np.std(r2))
    
plt.errorbar(np.unique(total), rMean, rStd, fmt='o')
plt.xlabel('#Ocorrencias')
plt.ylabel('%Level 2')

ax = sns.regplot(x=np.unique(total), y=rMean, ci=0, label='Regressao Linear')
ax.legend()

reinc = []
for secName in sectorsGeo['name']:
    civis, count = np.unique(data[data['Sector']==secName]['Subject_ID'], return_counts=True)
    reinc.append(float(len(civis[count != 1]))/len(civis))
sectorsGeo['reinc'] = reinc


fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.subplots_adjust(hspace=0.2, wspace=0.1)

ax = sectorsGeo.plot(ax=axes.flat[0], column='level2', cmap='YlGn', legend=True)
ax.set_title('Level 2')
ax.set_axis_off()
for idx, row in sectorsGeo.iterrows():
    ax.annotate(s=row['name'], xy=row['geometry'].centroid.coords[0],horizontalalignment='center')
    
ax = sectorsGeo.plot(ax=axes.flat[1], column='reinc', cmap='YlGn', legend=True)
ax.set_title('Reincidencias')
ax.set_axis_off()
for idx, row in sectorsGeo.iterrows():
    ax.annotate(s=row['name'], xy=row['geometry'].centroid.coords[0],horizontalalignment='center')

ax = sns.regplot(x=sectorsGeo['level2'], y=sectorsGeo['reinc'], ci=95)
ax.set_xlabel('% Level 2')
ax.set_ylabel('% Reincidencia')
ax.text(0.15, 0.385, 'W')

civis = np.unique(data['Subject_ID'])
civis = np.delete(civis, np.where(civis == 1708)) # Remove outlier

total = []
razao = []
for cName in civis:
    v = data[data['Subject_ID']==cName]['Incident_Type'].value_counts()
    if (not ('Level 2 - Use of Force' in v.keys())):        
        v['Level 2 - Use of Force'] = 0
    razao.append(float(v['Level 2 - Use of Force'])/np.sum(v))
    total.append(np.sum(v))

rMean = []
rStd = []
for t in np.unique(total):
    i = np.where(total == t)[0]
    r2 = np.array(razao)[i]
    rMean.append(np.mean(r2))
    rStd.append(np.std(r2))
    
plt.errorbar(np.unique(total), rMean, rStd, fmt='o')
plt.xlabel('#Ocorrencias')
plt.ylabel('%Level 2')

ax = sns.regplot(x=np.unique(total), y=rMean, ci=0, label='Regressao Linear')
ax.legend()