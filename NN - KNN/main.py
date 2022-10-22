from nn import nn
from knn import knn
from conjuntoDadosIrisFisher import tabela

vetor1 = [6.3, 5.0, 4.1, 2.1] # Iris versicolor
vetor2 = [4.3, 3.0, 1.1, 0.1] # Iris setosa
vetor3 = [6.7, 3.0, 5.2, 2.3] # Iris virginica
vetor4 = [5.7, 2.6, 3.5, 1.0] # Iris versicolor

print("- Executando o algoritmo NN:")
print("Testando uma Iris versicolor -> ", nn(vetor1, tabela))
print("Testando uma Iris setosa -> ", nn(vetor2, tabela))
print("Testando uma Iris virginica -> ", nn(vetor3, tabela))
print("Testando uma Iris versicolor -> ", nn(vetor4, tabela))

print("\n- Executando o algoritmo KNN:")
print("Testando uma Iris versicolor -> ", knn(5, vetor1, tabela))
print("Testando uma Iris setosa -> ", knn(5, vetor2, tabela))
print("Testando uma Iris virginica -> ", knn(5, vetor3, tabela))
print("Testando uma Iris versicolor -> ", knn(5, vetor4, tabela))
