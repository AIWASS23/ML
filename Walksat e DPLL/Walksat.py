import random
from gerador import literais
from DPLL import substitui
from copy import deepcopy

def valoraClausulas(expressao, modelo):
  sol=[]
  lit = literais(expressao)
  claus = expressao.split("and")
  for s, val in zip(lit,modelo):
    for i,c in enumerate(claus):
      c = c.replace(s, str(val))
      claus[i] = c
  return [eval(i) for i in claus]

def walksat(exps, probabilidade, trocaMaxima):
  claus = exps.split("and")
  lits = list(set([j for i in [literais(i) for i in claus] for j in i]))
  model = [random.choice([True, False]) for i in range(len(lits))]
  
  for i in range(trocaMaxima):
    valorClausulas = valoraClausulas(exps, model)

    if eval(substitui(claus, model)):
      return True, model, claus, valorClausulas
    
    nsClaus = [i for i,c in enumerate(valorClausulas) if c == False]
    simbolosNaClausula = []
    for i in nsClaus:
      simbolosNaClausula += literais(claus[i])
    if random.randint(1,100) > probabilidade:
      simboloTrocarValor = random.choice(simbolosNaClausula)
    else:
      simboloTrocarValor = None
      numerosVerdadeiros = valorClausulas.count(True)
      for simb in simbolosNaClausula:
        posicaoSimb = lits.index(simb)
        novoModelo = deepcopy(model)
        novoModelo[posicaoSimb] = not novoModelo[posicaoSimb]
        novoValorClausulas = valoraClausulas(exps, novoModelo)
        contagemVerdadeiros = novoValorClausulas.count(True)
        if contagemVerdadeiros > numerosVerdadeiros:
          numerosVerdadeiros = contagemVerdadeiros
          simboloTrocarValor = simb
    if simboloTrocarValor:
      posicaoFinal = lits.index(simboloTrocarValor)
      model[posicaoFinal] = not model[posicaoFinal]
  return False,model,claus,valorClausulas
