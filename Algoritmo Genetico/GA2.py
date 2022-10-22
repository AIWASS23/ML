def equacao1(x, y, z, w):
  valor = pow(x, 2) + pow(y, 3) + pow(z, 4) - pow(w, 5)
  erro = abs(valor)
  return erro

def equacao2(x, z, w):
  valor = pow(x, 2) + 3 * pow(z, 2) - w
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

  erro = equacao1(x, y, z, w)
  erro += equacao2(x, z, w)
  erro += equacao3(y, z)
  erro += equacao4(x, y, z, w)

  return erro

def fitness(erro):
  valor = 1 / (1 + erro)
  return valor

v1 = [0, 0, 0, 0]
v2 = [1, 1, 1, 1]

def get_bin(x):
    return format(x, 'b').zfill(4)

a = get_bin(25)
print(a)

print(fitness(erroTotal(v1)))
print(fitness(erroTotal(v2)))
