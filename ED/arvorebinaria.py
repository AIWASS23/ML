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

class Vertice:

    def __init__(self, rotulo):
        self.rotulo = rotulo

    def __eq__(self, outro):
        return outro.rotulo == self.rotulo

    def __repr__(self):
        return self.rotulo

    def __hash__(self):
        return hash(self.rotulo)

from graphviz import Graph
import numpy as np

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
        g.render(directory='.', view = True) 
        return g

    def plotar_grafo(vertices, adjacencias, titulo):
        g = Graph(comment = titulo, strict = True)
        tam_x = len(adjacencias)
        tam_y = len(adjacencias[0])
        for i in vertices:
            g.node(i, i, fontsize = "10")
        for i in range(tam_x):
            for j in range(tam_y):
                if i != j and adjacencias[i][j] != np.inf:
                    g.edge(vertices[i], vertices[j], dir = "none", label = str(adjacencias[i][j]))
        g.render(directory='.', view = True)  
        return g

from itertools import permutations

familia = Grafo()

familia.adicionaVertice('FRANCISCO (AVÔ)')
familia.adicionaVertice('NENA (MÃE)')
familia.adicionaVertice('TIO SIVALDO')
familia.adicionaVertice('TIO NILDO')
familia.adicionaVertice('EDUARDO (PRIMO)')
familia.adicionaVertice('ISAAC (PRIMO)')
familia.adicionaVertice('HERMÓGENES (PRIMO)')
familia.adicionaVertice('MIGUEL (IRMÃO)')
familia.adicionaVertice('JÚNIOR (IRMÃO)')
familia.adicionaVertice('MARCELO (EU)')
familia.adicionaAresta('FRANCISCO (AVÔ)', 'NENA (MÃE)')
familia.adicionaAresta('FRANCISCO (AVÔ)', 'TIO SIVALDO')
familia.adicionaAresta('FRANCISCO (AVÔ)', 'TIO NILDO')
familia.adicionaAresta('NENA (MÃE)', 'MIGUEL (IRMÃO)')
familia.adicionaAresta('NENA (MÃE)', 'JÚNIOR (IRMÃO)')
familia.adicionaAresta('NENA (MÃE)', 'GABRIEL (EU)')
familia.adicionaAresta('TIO SIVALDO', 'EDUARDO (PRIMO)')
familia.adicionaAresta('TIO SIVALDO', 'ISAAC (PRIMO)')
familia.adicionaAresta('TIO NILDO', 'HERMÓGENES (PRIMO)')

membros = permutations(['FRANCISCO (AVÔ)', 'NENA (MÃE)', 'TIO SIVALDO', 'TIO NILDO', 'EDUARDO (PRIMO)', 'ISAAC (PRIMO)', 'HERMÓGENES (PRIMO)', 'MIGUEL (IRMÃO)', 'JÚNIOR (IRMÃO)', 'GABRIEL (EU)'], 2)

print("- Árvore genealógica:\n")
print(familia.montaGrafo())

print("\n- Conexões entre os membros da família:\n")
for i, j in membros:
  print(i, "está ligado a ", j, ' pelos indivíduos: ', dfs(familia, i, j), "\n")
