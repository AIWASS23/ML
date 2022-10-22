from conjuntoDadosIrisFisher import distanciaEuclidiana
  
def vizinhos(vetor, tabela):
  vizinhos = []
  for i in tabela:
    vizinhos.append(distanciaEuclidiana(vetor, i[:-1]))
  return vizinhos
  
def nn(vetor, tabela):
  distancias = vizinhos(vetor, tabela)
  posicao = distancias.index(min(distancias))
  return tabela[posicao][-1]

def calculaCentroides(tabela):
  classes = set(i[-1] for i in tabela)
  centroides = []
  for i in classes:
    ct = [0 for j in tabela[0]]
    n = 0
    for k in tabela:
      if k[-1] == i:
        n += 1
        for w in range(len(k) - 1):
          ct[w] += k[w]
    for z, j in enumerate(ct[:-1]):
      ct[z] = j/n
    ct[-1] = i
    centroides.append(ct)
  return centroides
