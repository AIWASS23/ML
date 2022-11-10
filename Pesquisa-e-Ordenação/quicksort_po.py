from random import randint
import matplotlib as mpl
import matplotlib.pyplot as plt
import timeit
import time
import sys

limit = 1000000
sys.setrecursionlimit(limit)

def partir(array, low, high):
     
    if array[low] > array[high]:
        array[low], array[high] = array[high], array[low]
         
    j = k = low + 1
    g, p, q = high - 1, array[low], array[high]
     
    while k <= g:
         
        if array[k] < p:
            array[k], array[j] = array[j], array[k]
            j += 1
             
        elif array[k] >= q:
            while array[g] > q and k < g:
                g -= 1
                 
            array[k], array[g] = array[g], array[k]
            g -= 1
             
            if array[k] < p:
                array[k], array[j] = array[j], array[k]
                j += 1
                 
        k += 1
         
    j -= 1
    g += 1
     
    array[low], array[j] = array[j], array[low]
    array[high], array[g] = array[g], array[high]
    return j, g
 
def quickSortMelhorCaso(array, low, high):
     
    if low < high:

        leftPivot, rigthPivot = partir(array, low, high) 
        quickSortMelhorCaso(array, low, leftPivot - 1)
        quickSortMelhorCaso(array, leftPivot + 1, rigthPivot - 1)
        quickSortMelhorCaso(array, rigthPivot + 1, high)
        
def criar_Grafico(x, y, z, k):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Quick Sort')
    ax.set_ylabel('Tempo')
    ax.set_xlabel('Tamanho da Lista')
    ax.plot(x, y, label='Caso Aleatório')
    ax.plot(x, z, label='Pior Caso')
    ax.plot(x, k, label ='Melhor Caso')
    ax.legend()
    fig.savefig('graph.png')

def quickSort(array):

    L = []
    R = []
    tamanhoArray = len(array)
    if tamanhoArray < 2:
        return array
    pivo = array[len(array)//2]
    for i in array:
        if i < pivo:
            L.append(i)
        if i > pivo:
            R.append(i)
    
    return quickSort(L)+[pivo]+quickSort(R)

def quickSortPiorCaso(array):

    tamanhoArray = len(array)
    if tamanhoArray < 2:
        return array
    L = []
    R = []

    pivo = array[0]
    for i in range(1, len(array)):
        if array[i] <= pivo:
            L.append(array[i])
        if array[i] > pivo:
            R.append(array[i])
    return quickSortPiorCaso(L)+[pivo]+quickSortPiorCaso(R)
    
def gerarLista(tam):
    lista = []
    while len(lista) < tam:
        z = randint(1,1000000*tam)
        lista.append(z)
    return lista
      
def main():
    tamanhos = [100000, 200000, 300000, 400000, 500000, 800000, 1000000]
    tempo = []
    tempoPiorCaso = []
    tempoMelhorCaso = []

    for i in range(len(tamanhos)):

        lista = gerarLista(tamanhos[i])
        
        start = time.time()
        quickSort(lista)
        end = time.time()
        tempo.append(end - start)

        start = time.time()
        quickSortPiorCaso(lista)
        end = time.time()
        tempoPiorCaso.append(end - start)

        start = time.time()
        quickSortMelhorCaso(lista, 0, len(lista)-1)
        end = time.time()
        tempoMelhorCaso.append(end - start)

        stringInfo = 'Lista de tamanho {} - Caso aleatório: {:.4f}s; Pior Caso: {:.4f}s; Melhor Caso: {:.4f}s'
        print(stringInfo.format(tamanhos[i], tempo[i], tempoPiorCaso[i], tempoMelhorCaso[i]))

    criar_Grafico(tamanhos, tempo, tempoPiorCaso, tempoMelhorCaso)

if __name__ == '__main__':
    main()