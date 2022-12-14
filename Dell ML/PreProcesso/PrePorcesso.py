import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_table('fruit_data_with_colors_miss.txt')
data = pd.read_table('fruit_data_with_colors_miss.txt',na_values = ['.','?'])

data.fillna(0)
data.describe()
data.fillna(data.mean())
data['fruit_subtype'].value_counts().argmax()
data = data.fillna(data.mean())
data['fruit_subtype'] = data['fruit_subtype'].fillna(data['fruit_subtype'].value_counts().argmax())
data.shape[0]
data.isnull().sum()
data.isnull().sum()/data.shape[0] * 100
data = data[['fruit_label','fruit_name','fruit_subtype','width','height','color_score']]
data = data[['mass','width','height','color_score']]
data.describe()

mm = MinMaxScaler()
mm.fit(data)
data_escala = mm.transform(data)
macas = data[data['fruit_name'] == 'apple']
macas.describe()
est = macas['mass'].describe()
macas[(macas['mass'] > est['mean'] + (est['std']) * 2)]
macas[(macas['mass'] < est['mean'] - (est['std']) * 2)]

est['std']

