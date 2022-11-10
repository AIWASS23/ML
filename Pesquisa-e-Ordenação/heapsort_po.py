from random import randint
import matplotlib.pyplot as plt
import time

def heapify(arr, n, i):

    largest = i  
    l = 2 * i + 1     
    r = 2 * i + 2     
  
    if l < n and arr[i] < arr[l]:
        largest = l
  
    if r < n and arr[largest] < arr[r]:
        largest = r
  
    if largest != i:
        arr[i],arr[largest] = arr[largest],arr[i]
        heapify(arr, n, largest)

def heapSort(arr):
    n = len(arr)
  
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
  
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  
        heapify(arr, i, 0)
  
def criar_Grafico(x, y, z, k):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Heap Sort')
    ax.set_ylabel('Tempo')
    ax.set_xlabel('Tamanho da Lista')
    ax.plot(x, y, label='Caso Aleatório')
    ax.plot(x, z, label='Pior Caso')
    ax.plot(x, k, label ='Melhor Caso')
    ax.legend()
    fig.savefig('graph.png')

    
def gerarLista(tam):
    lista = []
    while len(lista) < tam:
        z = randint(1,100*tam)
        #if z not in lista: 
        lista.append(z)
    return lista
      
def main():
    tamanhos = [10000, 20000, 40000, 70000, 100000, 500000]
    tempo = []
    tempoPiorCaso = []
    tempoMelhorCaso = []

    for i in range(len(tamanhos)):

        lista = gerarLista(tamanhos[i])
        
        start = time.time()
        heapSort(lista)
        end = time.time()
        tempo.append(end - start)

        lista.reverse()

        start = time.time()
        heapSort(lista)
        end = time.time()
        tempoPiorCaso.append(end - start)

        start = time.time()
        heapSort(lista)
        end = time.time()
        tempoMelhorCaso.append(end - start)

        stringInfo = 'Lista de tamanho {} - Caso aleatório: {:.4f}s; Pior Caso: {:.4f}s; Melhor Caso: {:.4f}s'
        print(stringInfo.format(tamanhos[i], tempo[i], tempoPiorCaso[i], tempoMelhorCaso[i]))

    criar_Grafico(tamanhos, tempo, tempoPiorCaso, tempoMelhorCaso)

if __name__ == '__main__':
    main()