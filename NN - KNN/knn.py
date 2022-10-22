from conjuntoDadosIrisFisher import distanciaEuclidiana

def maioria(dicionario):
  return max(dicionario.items(), key = lambda x: x[1])
  
def knn(k, vetor, tabela):
  tabelaAux = sorted(tabela, key = lambda tup: distanciaEuclidiana(tup[:-1], vetor))
  classes = dict()
  for j in range(k):
    if tabelaAux[j][-1] in classes.keys():
      classes[tabelaAux[j][-1]] += 1
    else:
      classes[tabelaAux[j][-1]] = 1
  return maioria(classes)
