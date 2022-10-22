import random
import time
from DPLL import literais, geraString, DPLLsat
from Walksat import walksat
from gerador import geraConjuntoClausulas, desenhaGrafico

runtimeDPLL = []
runtimeWalkSAT = []
ratiomn = []

def execucao(qtdClausulas, qtdSimbolos):
  mn = qtdClausulas / qtdSimbolos
  ratiomn.append(mn)

  expressoes = geraConjuntoClausulas(qtdClausulas, qtdSimbolos)

  lits = set([j for i in [literais(i) for i in expressoes] for j in i])

  valoracao = [random.choice([True, False]) for k in range(len(lits))]

  expressao3FNC = geraString(expressoes)

  inicio = time.time()
  satistfacao = DPLLsat(expressao3FNC)
  fim = time.time()
  runtimeDPLL.append((fim - inicio)*1000)

  inicio = time.time()
  satistfacao = walksat(expressao3FNC,50,10000)
  fim = time.time()
  runtimeWalkSAT.append((fim - inicio)*1000)

contador = 1
for r1 in range(8):
  qtdClausulas = contador
  qtdSimbolos = 1

  execucao(qtdClausulas, qtdSimbolos)

  contador += 1


desenhaGrafico(ratiomn, runtimeDPLL, runtimeWalkSAT, 'DPLL-WalkSAT.png', 'Clause/symbol ratio m/n', 'Runtime')
