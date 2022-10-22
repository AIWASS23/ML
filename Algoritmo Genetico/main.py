import random
import numpy as np

def equacao1(x, y, z, w):
  valor = pow(x, 2) + pow(y, 3) + pow(z, 4) - pow(w, 5)
  erro = abs(valor)
  return erro

def equacao2(x, z, w):
  valor = pow(x, 2) + 3 * pow(z, 2) - w
  erro = abs(valor)
  return erro

def pequacao1(x, y):
  #valor = x + 2*y - 5
  #valor = 5*x - 4*y + 5
  valor = x + y - 10
  erro = abs(valor)
  return erro

def pequacao2(x, y):
  #valor = 3*x - 5*y - 4
  #valor = x + 2*y - 13
  valor = 2*x - y - 5
  erro = abs(valor)
  return erro

def equacao3(y, z):
  valor = pow(z, 5) - y - 10
  erro = abs(valor)
  return erro

def equacao4(x, y, z, w):
  valor = pow(x, 4) - z + y * w
  erro = abs(valor)
  return erro

def erroTotal(valores):
  x = valores[0]
  y = valores[1]
  z = valores[2]
  w = valores[3]
  '''
  erro = 0
  erro = pequacao1(x, y)
  erro += pequacao2(x, y)'''

  erro = equacao1(x, y, z, w)
  erro += equacao2(x, z, w)
  erro += equacao3(y, z)
  erro += equacao4(x, y, z, w)

  return erro

def randomizaBinario(tamanho):
  bits = [''] * tamanho

  for i in range(len(bits)):
    bits[i] = str(np.random.randint(2))
  
  strBits = ""
  for i in range(len(bits)):
    strBits += bits[i]

  valor = int(strBits, 2)
  
  return valor

class DNA:
  def __init__(self, objetivo, probabilidade_mutacao, qtd_individuos, qtd_selecionada, qtd_geracoes):
    self.objetivo = objetivo
    self.probabilidade_mutacao = probabilidade_mutacao
    self.qtd_individuos = qtd_individuos
    self.qtd_selecionada = qtd_selecionada
    self.qtd_geracoes = qtd_geracoes

  def cria_individuo(self, min = 0, max = 9):
    individuo = [np.random.randint(min, max) for _ in range(len(self.objetivo))]
    return individuo

  def cria_populacao(self):
    population = [self.cria_individuo() for _ in range(self.qtd_individuos)]
    return population

  def fitness(self, individuo):
    fitness = 1 / (1 + erroTotal(individuo))
    return fitness

  def seleciona(self, populacao):
    valores = [(self.fitness(i), i) for i in populacao]
    valores = [i[1] for i in sorted(valores)]

    selecionados = valores[len(valores)-self.qtd_selecionada:]

    return selecionados
      
  def reproduz(self, populacao, selecionados):
    divisor = 0
    pais = []

    for i in range(len(populacao)):
      divisor = np.random.randint(1, len(self.objetivo) - 1)
      pais = random.sample(selecionados, 2)

      populacao[i][:divisor] = pais[0][:divisor]
      populacao[i][divisor:] = pais[1][divisor:]
    
    return populacao

  
  def mutacao(self, populacao):
      
    for i in range(len(populacao)):
      if random.random() <= self.probabilidade_mutacao:
        divisor = np.random.randint(len(self.objetivo))
        novo_valor = randomizaBinario(4)

        while novo_valor == populacao[i][divisor]:
          novo_valor = randomizaBinario(4)
        
        populacao[i][divisor] = novo_valor

      return populacao
  
  def executa_algoritmo_genetico(self):
    populacao = self.cria_populacao()

    for i in range(self.qtd_geracoes):

      print('Geração: ', i)
      print('População', populacao)
      print()

      selecionados = self.seleciona(populacao)
      populacao = self.reproduz(populacao, selecionados)
      populacao = self.mutacao(populacao)
      
    print("Aproximação:", populacao[len(populacao) - 1])
  

objetivo = [0,0,0,0]
modelo = DNA(objetivo = objetivo, probabilidade_mutacao = 0.5, qtd_individuos = 500, 
qtd_selecionada = 10, qtd_geracoes = 50)
modelo.executa_algoritmo_genetico()
