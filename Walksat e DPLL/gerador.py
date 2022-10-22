import random
import matplotlib.pyplot as plt
import numpy as np

def geraClausulaAleatoria(qtdSimbolos):
  literais = 'ABCDEGHIJKLMNOPQRSUVWXYZ'
  literais = literais[:qtdSimbolos]
  clausula = ''

  #disjuncao = random.choice([True, False])
  disjuncao = True

  if disjuncao:
    #tam = np.random.randint(2, 4)
    tam = 3
    letras = ['']*tam
    
    for j in range(tam):
      letras[j] = literais[np.random.randint(qtdSimbolos)]
    
    clausula += geraOr(letras)+ ' and '

  else:
    letra = literais[np.random.randint(qtdSimbolos)]
    clausula += escolhePositivoNegativo(letra) + ' and '
    
  clausula = clausula[:len(clausula)-5]
  #clausula += '\n'
  #print(clausula)

  return clausula

def geraOr(letras):
  expressao = '('

  for i in range(len(letras)):
    expressao += escolhePositivoNegativo(letras[i]) + ' or '

  expressao = expressao[:len(expressao)-4]
  expressao += ')'

  return expressao

def escolhePositivoNegativo(literal):
  if random.choice([True, False]):
    return negar(literal)
    
  else:
    return literal

def negar(literal):
  return 'not(' + literal + ')'

def geraConjuntoClausulas(qtdClausulas, qtdSimbolos):
  #qtdClausulas = np.random.randint(1, 6)
  #print(qtdClausulas)
  #qtdClausulas = 1
  clausulas = ['a']*qtdClausulas

  for i in range(qtdClausulas):
    clausulas[i] = geraClausulaAleatoria(qtdSimbolos)

  return clausulas

def literais(expressao):
  atomos = []
  tks = expressao.split()
  for i in tks:
    i = i.replace('(','')
    i = i.replace(')','')
    i = i.replace('not','')
    if i in ('ABCDEGHIJKLMNOPQRSUVWXYZ') and i not in atomos:
      atomos.append(i)
  return atomos

def desenhaGrafico(x, y, y2, figura, xl="Entradas", yl="Saídas"):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.plot(x, y, label="WalkSAT")
    ax.plot(x, y2, label="DPLL")
    ax.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
    plt.ylabel(yl)
    plt.xlabel(xl)
    plt.show()
