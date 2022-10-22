from copy import deepcopy
from gerador import literais

def simboloPuro(clausulas, simbolo):
  pos = 0
  neg = 0

  for i in clausulas:
    if "not({})".format(simbolo) in i:
      neg += 1

    elif simbolo in i:
      pos += 1

  if neg > 0 and pos == 0:
    return True, 0
  elif pos > 0 and neg == 0:
    return True, 1
  else:
    return False, None

def clausulaUnitaria(clausulas, simbolo):
  for cl in clausulas:
    if "not({})".format(simbolo) in cl:
      n = 0
      for i in cl:
        if i in ("ABCDEGHIJKLMNOPQRSUVWXYZ"): n += 1
      if n == 1:
        return True, False
    else:
      for z in simbolo:
        if z in cl:
          n = 0
          for i in cl:
            if i in ("ABCDEGHIJKLMNOPQRSUVWXYZ"): n += 1
          if n == 1:
            return True, True
  return False, None

def substitui(claus, mod):
  exp = ""
  for i in range(len(claus)-1):
    exp = exp + claus[i] + " and "
  exp = exp + claus[-1]
  simb = literais(exp)
  nvexp = deepcopy(exp)
  for i,v in enumerate(mod):
    if simb[i] in nvexp:
      nvexp = nvexp.replace(simb[i],str(v))
  return nvexp

def geraString(claus):
  exp = ""
  for i in range(len(claus)-1):
    exp = exp + claus[i] + " and "
  exp = exp + claus[-1]
  return exp

def DPLL(c, s, md):
  if len(s) == 0:
    return eval(substitui(c, md)),md
  m = deepcopy(md)
  numClausulas = len(c)
  n = 0
  for i in c:
    sc = str(i)
    try:
      if eval(sc) == True: n+=1
      if eval(sc) == False: return False,m
    except(NameError):
      pass
  if n == numClausulas: return True,m
  primeiro = s[0]
  resto = s[1:]
  sp,val = simboloPuro(c,primeiro)
  if sp:
    if val:
      m.append(True)
    else:
      m.append(False)
    return DPLL(c,s[1:],m)
  cn, val = clausulaUnitaria(c,s)
  if cn:
    m.append(val)
    return DPLL(c,s[1:],m)
  mtt = deepcopy(m)
  mtt.append(True)
  rs,nm = DPLL(c,resto,mtt)
  if rs:
    return rs,nm
  mtt = deepcopy(m)
  mtt.append(False)
  rs,nm = DPLL(c,resto,mtt)
  if rs:
    return rs,nm
  return False, m

def DPLLsat(exps):
  clausulas = exps.split("and")
  lits = literais(exps)
  return DPLL(clausulas,lits,[])
