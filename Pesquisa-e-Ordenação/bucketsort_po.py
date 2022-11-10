from random import randint
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

def insertionSort(lista):
    for i in range (1, len (lista)):
        var = lista[i]
        j = i - 1
        while (j >= 0 and var < lista[j]):
            lista[j + 1] = lista[j]
            j = j - 1
        lista[j + 1] = var

def bucketSort(array):
    
    max_value = max(array)
    size = max_value/len(array)

    buckets_list= []
    for x in range(len(array)):
        buckets_list.append([]) 

    for i in range(len(array)):
        j = int (array[i] / size)
        if j != len (array):
            buckets_list[j].append(array[i])
        else:
            buckets_list[len(array) - 1].append(array[i])

    for z in range(len(array)):
        insertionSort(buckets_list[z])
            
    final_output = []
    for x in range(len (array)):
        final_output = final_output + buckets_list[x]
    return final_output

def bucketSortPiorCaso(bucket):
  insertionSort(bucket)
    
def criar_Grafico(x, y, z, k):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Bucket Sort')
    ax.set_ylabel('Tempo')
    ax.set_xlabel('Tamanho da Lista')
    ax.plot(x, y, label='Caso Aleatório')
    ax.plot(x, z, label='Melhor Caso')
    ax.plot(x, k, label ='Pior Caso')
    ax.legend()
    fig.savefig('graph.png')
    
def gerarLista(tam):
    lista = []
    while len(lista) < tam:
        z = randint(1,10*tam)
        #if z not in lista: 
        lista.append(z)
    return lista
      
def main():
    tamanhos = [10000, 20000, 30000, 40000, 50000, 80000, 100000]
    tempo = []
    tempoMelhorCaso = []
    tempoPiorCaso = []

    for i in range(len(tamanhos)):

        lista = gerarLista(tamanhos[i])
        
        start = time.time()
        bucketSort(lista)
        end = time.time()
        tempo.append(end - start)

        start = time.time()
        bucketSort(lista)
        end = time.time()
        tempoMelhorCaso.append(end - start)

        lista.reverse()

        start = time.time()
        bucketSort(lista)
        end = time.time()
        tempoPiorCaso.append(end - start)

        stringInfo = 'Lista de tamanho {} - Caso aleatório: {:.4f}s; Melhor Caso: {:.4f}s; Pior Caso: {:.4f}s;'
        print(stringInfo.format(tamanhos[i], tempo[i], tempoMelhorCaso[i], tempoPiorCaso[i]))

    criar_Grafico(tamanhos, tempo, tempoMelhorCaso, tempoPiorCaso)

if __name__ == '__main__':
    main()