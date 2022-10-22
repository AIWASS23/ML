from conjuntoDadosIrisFisher import tabela
from dmc import calculaCentroides, nn

centroides = calculaCentroides(tabela)
vetor1 = [6.3, 5.0, 4.1, 2.1] # Iris versicolor
vetor2 = [4.3, 3.0, 1.1, 0.1] # Iris setosa
vetor3 = [6.7, 3.0, 5.2, 2.3] # Iris virginica
vetor4 = [5.7, 2.6, 3.5, 1.0] # Iris versicolor

print("- Executando o algoritmo DMC:")
print("Testando uma Iris versicolor -> ", nn(vetor1, centroides))
print("Testando uma Iris setosa -> ", nn(vetor2, centroides))
print("Testando uma Iris virginica -> ", nn(vetor3, centroides))
print("Testando uma Iris versicolor -> ", nn(vetor4, centroides))

print("\n- Valores dos centroides:")
for i in calculaCentroides(tabela): print(i)
