from random import randint
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

def countingSort(Lista, place):
    n = len(Lista)
    output = [0 for i in range(0,n)]
  
    freq = [0 for i in range(0,10)]
  
    for i in range(0,n):
        freq[(Lista[i]//place)%10] += 1

    for i in range(1,10):
        freq[i] += freq[i - 1]      

    for i in range(n-1,-1,-1):
        output[freq[(Lista[i]//place)%10] - 1] = Lista[i] 
        freq[(Lista[i]//place)%10] -= 1

    for i in range(0,n): 
        Lista[i] = output[i] 

def radixSort(Lista):

    n = len(Lista)
    max = Lista[0]
  
    for i in Lista:
        if max < i:
            max = i
      
    place = 1
    while max/place > 0:
        countingSort(Lista, place)
        place *= 10  

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
        if z not in lista: 
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
        radixSort(lista)
        end = time.time()
        tempo.append(end - start)

        start = time.time()
        radixSort(lista)
        end = time.time()
        tempoMelhorCaso.append(end - start)

        lista.reverse()

        start = time.time()
        radixSort(lista)
        end = time.time()
        tempoPiorCaso.append(end - start)

        stringInfo = 'Lista de tamanho {} - Caso aleatório: {:.4f}s; Melhor Caso: {:.4f}s; Pior Caso: {:.4f}s;'
        print(stringInfo.format(tamanhos[i], tempo[i], tempoMelhorCaso[i], tempoPiorCaso[i]))

    criar_Grafico(tamanhos, tempo, tempoMelhorCaso, tempoPiorCaso)

if __name__ == '__main__':
    main()