from graphviz import Graph
from Vertice import Vertice

class Grafo:
  
  def __init__(self):
    self.listaArestas = dict()
    self.listaVertices = set()

  def __repr__(self):
    return str(self.listaArestas)

  def adicionaVertice(self, rotulo):
    self.listaVertices.add(Vertice(rotulo))

  def adicionaAresta(self, r1, r2):
    if not self.listaArestas.get(r1):
      self.listaArestas[r1] = [r2]
    else:
      self.listaArestas[r1].append(r2)

    if not self.listaArestas.get(r2):
      self.listaArestas[r2] = [r1]
    else:
      self.listaArestas[r2].append(r1)

  def procuraVertice(self, rotulo):
    for i in self.listaVertices:
      if i.rotulo == rotulo:
        return i
    return -1

  def montaGrafo(self):
    g = Graph(comment='Família', strict=True)
    for i in self.listaVertices:
      g.node(i.rotulo, i.rotulo, fontsize="10")
    for k, v in self.listaArestas.items():
      for j in v:
        g.edge(k, j, dir="none")
    return g
