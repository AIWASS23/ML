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
