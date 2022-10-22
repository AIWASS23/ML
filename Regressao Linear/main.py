import numpy as np
from regressaoLinear import exibirRegressaoLinear

X = np.random.randint(1,1000,20)
Y = np.random.randint(1,1000,20)

vetores = np.array([X,Y])

exibirRegressaoLinear(vetores)
