# ### - Métricas de Avaliação
# ### - Classificação
# ### - Acurácia
# ### - Regressão
# ### - Média absoluta do erro
# 
# ### Desvantagens destas métricas
# ### Como escolher a métrica mais adequada?
# 
# ### Aplicações diferentes tem objetivos diferentes
# ### Acurácia é bastante usada, mas existem outras métricas
# #### Número de buscas na web (satisfação do usuário)
# #### Retorno financeiro (comércio)
# #### Aumento na taxa de sobrevivência (saúde)
# 
# ### Dados desbalanceados
# ### Problema com duas classes
# ### Classe 0 - Irrelevante
# ### Classe 1 - Relevante
# 
# ### Exemplo
# ### Identificação de fraudes em cartões
# ### 1 em cada 1000 é relevante
# ### Acurácia = predições corretas / total de itens
# 
# #### Um classificador obteve 99,9% de acurácia
# #### Esse resultado é bom?
# #### Comparar com um classificador muito simples
# 
# #### Classificador Dummy
# #### Verificador de sanidade
# #### Baseline para classificação
# #### Tipos
# #### Mais frequente (most_frequent)
# #### Aleatório (uniform)
# #### Aleatório com distribuição igual aos dados de treino (stratified)
# #### Constante e configurado pelo usuário (constant)
# 
# #### O seu classificador pode estar ruim por erros em atributos, overfitting ou desbalanceamento das classes
# 
# #### Regressor dummy
# #### Tipos
# #### Média dos rótulos de treinamento (mean)
# #### Mediana das saídas do treinamento (median)
# #### Quartil das saídas de treinamento. 0 para mínimo, 0.5 para média e 1 para máximo (quantile)
# #### Constante e configurado pelo usuário (constant)

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.datasets import load_iris, load_boston, load_breast_cancer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score


X,y = load_iris(return_X_y = True)

dc = DummyClassifier(strategy = 'stratified')
dc.fit(X,y)
dc.score(X,y)

X,y = load_boston(return_X_y = True)

dr = DummyRegressor(strategy = 'mean')
dr.fit(X,y)
dr.score(X,y)

X,y = load_breast_cancer(return_X_y = True)

dc = DummyClassifier(strategy = 'stratified')
dc.fit(X,y)
confusion_matrix(y,dc.predict(X))

# #### Recall (Sensitividade)
# #### Útil para aplicações médicas
# #### Verdadeiros positivos / (Verdadeiros positivos + Falsos negativos)
# #### Precisão
# #### Útil para sistemas de recomendação
# #### Verdadeiros positivos / (Verdadeiros positivos + Falsos positivos)

print(classification_report(y,dc.predict(X)))

# ### Validação cruzada

X,y = load_iris(return_X_y = True) 
dc = DummyClassifier(strategy = 'stratified')
cross_val_score(dc, X, y, cv = 3)
cross_val_score(dc, X, y, cv = 3).mean()