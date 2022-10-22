def adjacenteNaoVisitado(visitados, lista):
  for i in lista:
    if i not in visitados:
      return i

def dfs(grafo, inicio, meta):
  fronteira = [inicio]
  visitados = set()

  while fronteira:
    v = fronteira[-1]

    if v == meta:
      return fronteira

    visitados.add(v)
    s = adjacenteNaoVisitado(visitados, grafo.listaArestas.get(v))

    if s:
      fronteira.append(s)
    else:
      fronteira.pop()

  return "A busca não foi bem sucedida.."
