
import graphviz
from graphviz import Graph
import numpy as np

class Noh:

    def __init__(self, dado=None):

        self.esquerdo = None
        self.direito = None
        self.dado = dado

    def __str__(self):
        
        return "{"+str(self.dado)+"}"

noh = Noh()
noh.dado = 2

print(noh)

class ArvoreBinaria: 
                   # Definição da classe árvore
    def __init__(self):
        self.raiz = None   
                 # inicializa a raiz
    def criaNoh(self, dado):        # cria um novo noh e o retorna
        return Noh(dado)

    def insere(self, raiz, dado): 
          # insere um novo dado
        if raiz == None:            # arvore vazia
            return self.criaNoh(dado)
        else:
            if dado <= raiz.dado:
                raiz.esquerdo = self.insere(raiz.esquerdo, dado)
            else:
                raiz.direito = self.insere(raiz.direito, dado)
        return raiz
       
    def pesquisa(self, raiz, valor): # Pesquisa um valor na árvore
        if raiz == None:
            return print(False)
        else:
            if valor == raiz.dado:
                return print(True)
            else:
                if valor < raiz.dado:
                    return self.pesquisa(raiz.esquerdo, valor)
                else:
                    return self.pesquisa(raiz.direito, valor)
    def imprimirArvore(self, raiz): # imprime a árvore
        if raiz == None:
            pass
        else:
            self.imprimirArvore(raiz.esquerdo)
            print("{",raiz.dado,"}", end=' ')
            self.imprimirArvore(raiz.direito)
    def imprimeArvoreInvertida(self, raiz): # imprime a árvore invertida
        if raiz == None:
            pass
        else:
            self.imprimeArvoreInvertida(raiz.direito)
            print("{",raiz.dado,"}", end=' ')
            self.imprimeArvoreInvertida(raiz.esquerdo)

    def imprimeNohs(self,raiz):
        if raiz == None: return
        a = raiz.dado
        if raiz.esquerdo != None:
            b = raiz.esquerdo.dado
        else:
            b = None
        if raiz.direito != None:
            c = raiz.direito.dado
        else:
            c = None
        print("{",a,"[",b,",",c,"]","}", end=' ')
        self.imprimeNohs(raiz.esquerdo)
        self.imprimeNohs(raiz.direito)

arvore = ArvoreBinaria()

raiz = arvore.criaNoh(2)

arvore.insere(raiz, 2)
arvore.insere(raiz, 4)
arvore.insere(raiz, 8)
arvore.insere(raiz, 16)
arvore.insere(raiz, 32)
arvore.insere(raiz, 64)
arvore.insere(raiz, 128)
arvore.insere(raiz, 256)
arvore.insere(raiz, 512)

arvore.imprimirArvore(raiz)
print("\n")
arvore.imprimeArvoreInvertida(raiz)
print("\n")
arvore.imprimeNohs(raiz)
print("\n")
arvore.pesquisa(raiz, 23)
print("\n")
arvore.pesquisa(raiz, 32)

v1 = "a"
v2 = "b"
v3 = "c"

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

g = Graph(comment = "Teste")
g.node(v1,v2,fontsize = "10")
g.edge
g

e = graphviz.Graph('ER', filename='er.gv', engine='neato')

e.attr('node', shape='box')
e.node('course')
e.node('institute')
e.node('student')

e.attr('node', shape='ellipse')
e.node('name0', label='name')
e.node('name1', label='name')
e.node('name2', label='name')
e.node('code')
e.node('grade')
e.node('number')

e.attr('node', shape='diamond', style='filled', color='lightgrey')
e.node('C-I')
e.node('S-C')
e.node('S-I')

e.edge('name0', 'course')
e.edge('code', 'course')
e.edge('course', 'C-I', label='n', len='1.00')
e.edge('C-I', 'institute', label='1', len='1.00')
e.edge('institute', 'name1')
e.edge('institute', 'S-I', label='1', len='1.00')
e.edge('S-I', 'student', label='n', len='1.00')
e.edge('student', 'grade')
e.edge('student', 'name2')
e.edge('student', 'number')
e.edge('student', 'S-C', label='m', len='1.00')
e.edge('S-C', 'course', label='n', len='1.00')

e.attr(label=r'\n\nEntity Relation Diagram\ndrawn by NEATO')
e.attr(fontsize='20')

e.view()
e

u = graphviz.Digraph('unix', filename='unix.gv',
                     node_attr={'color': 'lightblue2', 'style': 'filled'})
u.attr(size='6,6')

u.edge('5th Edition', '6th Edition')
u.edge('5th Edition', 'PWB 1.0')
u.edge('6th Edition', 'LSX')
u.edge('6th Edition', '1 BSD')
u.edge('6th Edition', 'Mini Unix')
u.edge('6th Edition', 'Wollongong')
u.edge('6th Edition', 'Interdata')
u.edge('Interdata', 'Unix/TS 3.0')
u.edge('Interdata', 'PWB 2.0')
u.edge('Interdata', '7th Edition')
u.edge('7th Edition', '8th Edition')
u.edge('7th Edition', '32V')
u.edge('7th Edition', 'V7M')
u.edge('7th Edition', 'Ultrix-11')
u.edge('7th Edition', 'Xenix')
u.edge('7th Edition', 'UniPlus+')
u.edge('V7M', 'Ultrix-11')
u.edge('8th Edition', '9th Edition')
u.edge('1 BSD', '2 BSD')
u.edge('2 BSD', '2.8 BSD')
u.edge('2.8 BSD', 'Ultrix-11')
u.edge('2.8 BSD', '2.9 BSD')
u.edge('32V', '3 BSD')
u.edge('3 BSD', '4 BSD')
u.edge('4 BSD', '4.1 BSD')
u.edge('4.1 BSD', '4.2 BSD')
u.edge('4.1 BSD', '2.8 BSD')
u.edge('4.1 BSD', '8th Edition')
u.edge('4.2 BSD', '4.3 BSD')
u.edge('4.2 BSD', 'Ultrix-32')
u.edge('PWB 1.0', 'PWB 1.2')
u.edge('PWB 1.0', 'USG 1.0')
u.edge('PWB 1.2', 'PWB 2.0')
u.edge('USG 1.0', 'CB Unix 1')
u.edge('USG 1.0', 'USG 2.0')
u.edge('CB Unix 1', 'CB Unix 2')
u.edge('CB Unix 2', 'CB Unix 3')
u.edge('CB Unix 3', 'Unix/TS++')
u.edge('CB Unix 3', 'PDP-11 Sys V')
u.edge('USG 2.0', 'USG 3.0')
u.edge('USG 3.0', 'Unix/TS 3.0')
u.edge('PWB 2.0', 'Unix/TS 3.0')
u.edge('Unix/TS 1.0', 'Unix/TS 3.0')
u.edge('Unix/TS 3.0', 'TS 4.0')
u.edge('Unix/TS++', 'TS 4.0')
u.edge('CB Unix 3', 'TS 4.0')
u.edge('TS 4.0', 'System V.0')
u.edge('System V.0', 'System V.2')
u.edge('System V.2', 'System V.3')

u.view()
u

s = graphviz.Digraph('structs', filename='structs.gv',
                     node_attr={'shape': 'plaintext'})

s.node('struct1', '''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
  <TR>
    <TD>left</TD>
    <TD PORT="f1">middle</TD>
    <TD PORT="f2">right</TD>
  </TR>
</TABLE>>''')
s.node('struct2', '''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
  <TR>
    <TD PORT="f0">one</TD>
    <TD>two</TD>
  </TR>
</TABLE>>''')
s.node('struct3', '''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
  <TR>
    <TD ROWSPAN="3">hello<BR/>world</TD>
    <TD COLSPAN="3">b</TD>
    <TD ROWSPAN="3">g</TD>
    <TD ROWSPAN="3">h</TD>
  </TR>
  <TR>
    <TD>c</TD>
    <TD PORT="here">d</TD>
    <TD>e</TD>
  </TR>
  <TR>
    <TD COLSPAN="3">f</TD>
  </TR>
</TABLE>>''')

s.edges([('struct1:f1', 'struct2:f0'), ('struct1:f2', 'struct3:here')])

s.view()
s

s = graphviz.Digraph('structs', filename='structs_revisited.gv',
                     node_attr={'shape': 'record'})

s.node('struct1', '<f0> left|<f1> middle|<f2> right')
s.node('struct2', '<f0> one|<f1> two')
s.node('struct3', r'hello\nworld |{ b |{c|<here> d|e}| f}| g | h')

s.edges([('struct1:f1', 'struct2:f0'), ('struct1:f2', 'struct3:here')])

s.view()
s

import graphviz
from graphviz import nohtml

g = graphviz.Digraph('g', filename='btree.gv',
                     node_attr={'shape': 'record', 'height': '.1'})

g.node('node0', nohtml('<f0> |<f1> G|<f2>'))
g.node('node1', nohtml('<f0> |<f1> E|<f2>'))
g.node('node2', nohtml('<f0> |<f1> B|<f2>'))
g.node('node3', nohtml('<f0> |<f1> F|<f2>'))
g.node('node4', nohtml('<f0> |<f1> R|<f2>'))
g.node('node5', nohtml('<f0> |<f1> H|<f2>'))
g.node('node6', nohtml('<f0> |<f1> Y|<f2>'))
g.node('node7', nohtml('<f0> |<f1> A|<f2>'))
g.node('node8', nohtml('<f0> |<f1> C|<f2>'))

g.edge('node0:f2', 'node4:f1')
g.edge('node0:f0', 'node1:f1')
g.edge('node1:f0', 'node2:f1')
g.edge('node1:f2', 'node3:f1')
g.edge('node2:f2', 'node8:f1')
g.edge('node2:f0', 'node7:f1')
g.edge('node4:f2', 'node6:f1')
g.edge('node4:f0', 'node5:f1')

g.view()
g

t = graphviz.Digraph('TrafficLights', filename='traffic_lights.gv',
                     engine='neato')

t.attr('node', shape='box')
for i in (2, 1):
    t.node(f'gy{i:d}')
    t.node(f'yr{i:d}')
    t.node(f'rg{i:d}')

t.attr('node', shape='circle', fixedsize='true', width='0.9')
for i in (2, 1):
    t.node(f'green{i:d}')
    t.node(f'yellow{i:d}')
    t.node(f'red{i:d}')
    t.node(f'safe{i:d}')

for i, j in [(2, 1), (1, 2)]:
    t.edge(f'gy{i:d}', f'yellow{i:d}')
    t.edge(f'rg{i:d}', f'green{i:d}')
    t.edge(f'yr{i:d}', f'safe{j:d}')
    t.edge(f'yr{i:d}', f'red{i:d}')
    t.edge(f'safe{i:d}', f'rg{i:d}')
    t.edge(f'green{i:d}', f'gy{i:d}')
    t.edge(f'yellow{i:d}', f'yr{i:d}')
    t.edge(f'red{i:d}', f'rg{i:d}')

t.attr(overlap='false')
t.attr(label=r'PetriNet Model TrafficLights\n'
             r'Extracted from ConceptBase and layed out by Graphviz')
t.attr(fontsize='12')

t.view()
t

g = graphviz.Graph('G', filename='fdpclust.gv', engine='fdp')

g.node('e')

with g.subgraph(name='clusterA') as a:
    a.edge('a', 'b')
    with a.subgraph(name='clusterC') as c:
        c.edge('C', 'D')

with g.subgraph(name='clusterB') as b:
    b.edge('d', 'f')

g.edge('d', 'D')
g.edge('e', 'clusterB')
g.edge('clusterC', 'clusterB')

g.view()
g

g = graphviz.Digraph('G', filename='cluster_edge.gv')
g.attr(compound='true')

with g.subgraph(name='cluster0') as c:
    c.edges(['ab', 'ac', 'bd', 'cd'])

with g.subgraph(name='cluster1') as c:
    c.edges(['eg', 'ef'])

g.edge('b', 'f', lhead='cluster1')
g.edge('d', 'e')
g.edge('c', 'g', ltail='cluster0', lhead='cluster1')
g.edge('c', 'e', ltail='cluster0')
g.edge('d', 'h')

g.view()
g

g = graphviz.Graph('G', filename='g_c_n.gv')
g.attr(bgcolor='purple:pink', label='agraph', fontcolor='white')

with g.subgraph(name='cluster1') as c:
    c.attr(fillcolor='blue:cyan', label='acluster', fontcolor='white',
           style='filled', gradientangle='270')
    c.attr('node', shape='box', fillcolor='red:yellow',
           style='filled', gradientangle='90')
    c.node('anode')

g.view()
g

g = graphviz.Digraph('G', filename='angles.gv')
g.attr(bgcolor='blue')

with g.subgraph(name='cluster_1') as c:
    c.attr(fontcolor='white')
    c.attr('node', shape='circle', style='filled', fillcolor='white:black',
           gradientangle='360', label='n9:360', fontcolor='black')
    c.node('n9')
    for i, a in zip(range(8, 0, -1), range(360 - 45, -1, -45)):
        c.attr('node', gradientangle=f'{a:d}', label=f'n{i:d}:{a:d}')
        c.node(f'n{i:d}')
    c.attr(label='Linear Angle Variations (white to black gradient)')

with g.subgraph(name='cluster_2') as c:
    c.attr(fontcolor='white')
    c.attr('node', shape='circle', style='radial', fillcolor='white:black',
           gradientangle='360', label='n18:360', fontcolor='black')
    c.node('n18')
    for i, a in zip(range(17, 9, -1), range(360 - 45, -1, -45)):
        c.attr('node', gradientangle=f'{a:d}', label=f'n{i:d}:{a:d}')
        c.node(f'n{i:d}')
    c.attr(label='Radial Angle Variations (white to black gradient)')

g.edge('n5', 'n14')

g.view()
g

d = graphviz.Digraph(filename='rank_same.gv')

with d.subgraph() as s:
    s.attr(rank='same')
    s.node('A')
    s.node('X')

d.node('C')

with d.subgraph() as s:
    s.attr(rank='same')
    s.node('B')
    s.node('D')
    s.node('Y')

d.edges(['AB', 'AC', 'CD', 'XY'])

d.view()
g

g = graphviz.Graph(filename='colors.gv')

red, green, blue = 64, 224, 208
assert f'#{red:x}{green:x}{blue:x}' == '#40e0d0'

g.node('RGB: #40e0d0', style='filled', fillcolor='#40e0d0')

g.node('RGBA: #ff000042', style='filled', fillcolor='#ff000042')

g.node('HSV: 0.051 0.718 0.627', style='filled', fillcolor='0.051 0.718 0.627')

g.node('name: deeppink', style='filled', fillcolor='deeppink')

g.view()
g

