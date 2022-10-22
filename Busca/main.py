from itertools import permutations
from Grafo import Grafo
from depthFirstSearch import dfs

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
familia.adicionaVertice('GABRIEL (EU)')

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
