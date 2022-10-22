import matplotlib.pyplot as plt
import numpy as np

def exibirRegressaoLinear(vetores):
  Xmin = min(vetores[0])
  Xmax = max(vetores[0])

  a = np.linspace(Xmin,Xmax)
  theta = minimosQuadradosLinear(vetores)
  b = theta[0][0] * a + theta[1][0]

  plt.scatter(vetores[0],vetores[1])
  plt.plot(a,b)
  plt.show()

def minimosQuadradosLinear(vetores):
  m,n = vetores.shape
  cx = vetores[0].reshape(n,1)
  c2 = np.ones(len(cx)).reshape(n,1)
  A = np.hstack((cx,c2))

  Y = vetores[1].reshape(n,1)

  atainv = np.linalg.inv(np.matmul(A.T,A))
  aty = np.matmul(A.T,Y)
  theta = np.matmul(atainv,aty)

  return theta
