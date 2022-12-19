import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('diabetes.csv')
df.head()
df.describe()

plt.plot(df['Age'], df['Pregnancies'], 'o')
plt.xlabel('Idade')
plt.ylabel('Gravidez')
plt.show()

y = df['Outcome']
x = df.drop('Outcome', axis = 1)
x.head()

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.2)

modelo = LogisticRegression(max_iter = 5000)
modelo.fit(x_treino, y_treino)

y_previsto = modelo.predict(x_teste)
print(y_previsto[:10])

accuracy = accuracy_score(y_teste, y_previsto)
print(accuracy * 100.)