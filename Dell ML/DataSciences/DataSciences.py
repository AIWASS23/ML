
import pandas as pd
import matplotlib.pyplot as plt

frutas = pd.read_table('fruit_data_with_colors.txt',sep='\t')

frutas.shape
frutas.head(5)
frutas.describe()
frutas.describe()['mass']
frutas.describe()['mass']['min']
frutas['mass']
frutas[['mass','color_score']]
frutas[10:15]

i = 15
frutas[i-5:i]
frutas[i:i+5]

frutas[['mass','color_score']][i:i+5]
freq = frutas['fruit_name'].value_counts()

freq.plot(kind='bar')
plt.show()

frutas['fruit_name'] == 'apple'
macas = frutas['fruit_name'] == 'apple'
frutas[macas]
frutas['mass'] > 175
frutas[macas & frutas['mass'] > 175]
pesadas = frutas['mass'] > 175
frutas[macas & pesadas]
X1 = frutas[macas & pesadas]['width']
X2 = frutas[macas & pesadas]['height']

plt.scatter(X1,X2)
plt.show()

plt.scatter(X1,X2)
plt.xlabel('comprimento')
plt.ylabel('altura')
plt.show()

X1 = frutas['width']
X2 = frutas['height']

plt.scatter(X1,X2)
plt.xlabel('comprimento')
plt.ylabel('altura')
plt.show()

y = frutas['fruit_label']

plt.scatter(X1,X2,c=y)
plt.xlabel('comprimento')
plt.ylabel('altura')
plt.show()



