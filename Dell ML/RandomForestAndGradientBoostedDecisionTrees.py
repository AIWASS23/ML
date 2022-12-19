# # Random Forest e Gradient Boosted Decision Trees
#
# ### Modelo de Gradient Boosting
#
# ### Conjunto de modelos é criado de modo que cada modelo tenta melhorar a performance de um modelo anterior
# ### Modelos baseados nas Árvores de Decisão (Decision Trees)
# ### Árvores de Decisão são interpretáveis
# ### Sequência de regras "Se-Então"
# ### Propensas a overfit
# 
# ### Evitam o overfit natural das Árvores de Decisão
# ### Uso de weak learners (aprendizado lento ou fraco)
# ### Modelos simples coordenados para construir um modelo complexo
#
# ### Random Forest
#
# ### Utilização de árvores em paralelo
#
# ### Gradient Boosted Decision Trees
#
# ### Utilização de árvores em série
#
# ### Random Forest
#
# ### Paralelização de árvores de decisão
# ### Divisão do conjunto de dados aleatoriamente em linhas e colunas
# ### Por exemplo, tomar um conjunto de dados com 100 linhas (indivíduos) e 10 colunas (atributos)
# ### Um exemplo de divisão seriam 2 conjuntos de 100 linhas e 5 colunas
# ### Outro exemplo de divisão seriam 2 conjuntos de 50 linhas e 10 colunas
# ### As divisões podem ser totalmente aleatórias
# ### Árvores de Decisão comuns atuavam sobre o conjunto completo dos dados, apresentando regras que dividiam perfeitamente os mesmos, podendo causar overfit
# ### Nas Random Forests, cada árvore é responsável por uma parte do conjunto de dados, sendo impossível que uma árvore divida perfeitamente a totalidade dos mesmos, criando um overfit local, que não influi no cenário total representado pelos dados
# ### O resultado é agregado utilizando um método de votação
# ### Em um problema de classificação, escolhemos a classe eleita pela maioria das árvores
# ### Em um problema de regressão, escolhemos a média dos resultados obtidos pelas árvores
# ### Teorema: Se mais da metade das árvores decidem por uma resposta, esta resposta é a correta

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split

X,y = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X,y)

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train,y_train)
rf.score(X_test,y_test)

X,y = load_boston(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X,y)
rf = RandomForestRegressor(n_estimators = 100)
rf.fit(X_train,y_train)
rf.score(X_test,y_test)

X,y = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X,y)
gb = GradientBoostingClassifier(n_estimators = 200,max_depth = 3)
gb.fit(X_train,y_train)
gb.score(X_test,y_test)

X,y = load_boston(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X,y)
gb = GradientBoostingRegressor(n_estimators = 500,max_depth = 3)
gb.fit(X_train,y_train)
gb.score(X_test,y_test)


# ### Considerações Finais
# ##### Random Forest e Gradient Boosted Decision Trees
# ##### São técnicas baseadas em conjuntos de modelos (weak learners)
# ##### Random Forests atuam bem com dados com outliers
# ##### Gradient Boosted Decision Trees funcionam bem com dados esparsos