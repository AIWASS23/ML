# ### Principal Component Analysis

# ### Encontrar a aproximação do conjunto de dados original em um espaço de dimensão reduzido
#
# ### Objetivos:
#
# ### Visualização dos dados
# ### Redução da complexidade de modelos de Aprendizado de Máquina
# ### Uma das principais informações utilizadas será a variância dos dados
# ### Transformação linear
# ### Novos atributos são uma combinação linear dos atributos originais
# 
# ### Passos do PCA
#
# ### Toma os dados de entrada
# ### Subtração das médias e divisão pelo desvio padrão - Padronização (Standardization)
# ### Cálculo da matriz de covariância
# ### Cálculo dos autovalores e autovetores
# ### Seleção dos k maiores autovalores
# ### Multiplicação da transposta dos k autovetores pela transposta do conjunto de dados original padronizado (standardized)
# ### Novo conjunto de dados

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import load_boston

X,y = load_boston(return_X_y = True)
X.shape

pca = PCA(n_components = 10)
X_pca = pca.fit_transform(X)
X_pca.shape

plt.scatter(X_pca[:,0], X_pca[:,1])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_pca,y)

lr = LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)