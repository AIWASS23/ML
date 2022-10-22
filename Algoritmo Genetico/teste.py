import numpy as np
import random as rd

tamCromossomo = 30
pc = 0.95
pm = 0.1
numgeracoes = 100
tamPopulacao = 50

p = np.zeros((tamPopulacao, tamCromossomo))
for i in range(tamPopulacao):
  for j in range(tamCromossomo):
    a = rd.uniform(0,1)
    if (a >= 0.5):
      p[i][j] = 1
    else:
      p[i][j] = 0

ind = np.zeros(tamCromossomo)
individuo = np.zeros(tamPopulacao)
Aptidao = np.zeros(tamPopulacao)
novageracao = np.zeros((tamPopulacao, tamCromossomo))
geracoes = 0

while(geracoes <= numgeracoes):
  novosindividuos = 0
  while(novosindividuos < (tamPopulacao - 1)):
    for i in range(tamPopulacao):
      ind[:] = p[i, :]
      conv = 0
      for j in range(tamCromossomo):
        conv = conv + ind[j]*(2**(tamCromossomo-(j+1)))
      individuo[i] = (512/(2**tamCromossomo-1))*conv
    
    TotalAptidao = 0
    for i in range(tamPopulacao):
      Aptidao[i] = abs(individuo[i]*np.sin(np.sqrt(abs(individuo[i]))))
      TotalAptidao = Aptidao[i]+TotalAptidao

    pic = np.zeros(tamPopulacao)
    pitotal = np.zeros(tamPopulacao)
    pic = (1/TotalAptidao)*Aptidao

    for i in range(tamPopulacao):
      if(i == 0):
        pitotal[i] = pic[i]
      else:
        pitotal[i] = pic[i] + pitotal[i-1]

    roleta1 = rd.uniform(0,1)
    i=0
    while(roleta1 > pitotal[i]):
      i += 1
    
    pai1 = i

    roleta2 = rd.uniform(0,1)
    i=0
    while(roleta2 > pitotal[i]):
      i += 1
    
    pai2 = i

    while(pai2 == pai1):
      roleta2 = rd.uniform(0,1)
      i=0
      while(roleta2 > pitotal[i]):
        i += 1
        pai2 = i
    
    if(pc > rd.uniform(0,1)):
      c = round(1+(tamCromossomo-2)*rd.uniform(0,1))
      gene11 = p[pai1][0:c]
      gene12 = p[pai1][c:tamCromossomo]
      gene21 = p[pai2][0:c]
      gene22 = p[pai2][c:tamCromossomo]
      filho1 = np.concatenate((gene11, gene22), axis=None)
      filho2 = np.concatenate((gene21, gene12), axis=None)

      novageracao[novosindividuos, :] = filho1
      novosindividuos = novosindividuos + 1
      novageracao[novosindividuos, :] = filho2
      novosindividuos = novosindividuos + 1
    
    if(pm > rd.uniform(0,1)):
      d = round(1+(tamCromossomo-2)*rd.uniform(0,1))
      if(novageracao[novosindividuos-2][d] == 0):
        novageracao[novosindividuos-2][d] = 1
      else:
        novageracao[novosindividuos-2][d] = 0
      if(novageracao[novosindividuos-1][d] == 0):
        novageracao[novosindividuos-1][d] = 1
      else:
        novageracao[novosindividuos-1][d] = 0
  
  indice = Aptidao.argmax()
  elem = individuo[indice]
  print(elem)
  p = novageracao
  geracoes = geracoes+1
