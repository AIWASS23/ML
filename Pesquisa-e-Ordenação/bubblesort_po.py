from random import randint
import matplotlib as mpl
import matplotlib.pyplot as plt
import time


def criar_Grafico(x, y, z, k):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Bubble')
    ax.set_ylabel('Tempo')
    ax.set_xlabel('Tamanho da Lista')
    ax.plot(x, y, label='Caso Aleatório')
    ax.plot(x, z, label='Pior Caso')
    ax.plot(x, k, label ='Melhor Caso')
    ax.legend()
    fig.savefig('graph.png')

def bubbleSort(V):  

    for j in range(0,len(V)):
        for i in range(0,len(V)-1):
            if V[i]>V[i+1]:
                aux = V[i+1]
                V[i+1] = V[i]
                V[i] = aux

def gerarLista(tam):
    lista = []
    while len(lista) < tam:
        z = randint(1,1*tam)
        if z not in lista: 
            lista.append(z)
    return lista
      
def main():

    tamanhos = [1000, 2000, 3000, 4000, 5000, 8000, 11000, 15000]
    tempo = []
    tempoPiorCaso = []
    tempoMelhorCaso = []

    for i in range(len(tamanhos)):

        lista = gerarLista(tamanhos[i])
        
        start = time.time()
        bubbleSort(lista)
        end = time.time()
        tempo.append(end - start)

        lista.reverse()

        start = time.time()
        bubbleSort(lista)
        end = time.time()
        tempoPiorCaso.append(end - start)

        start = time.time()
        bubbleSort(lista)
        end = time.time()
        tempoMelhorCaso.append(end - start)

        stringInfo = 'Lista de tamanho {} - Caso aleatório: {:.4f}s; Pior Caso: {:.4f}s; Melhor Caso: {:.4f}s'
        print(stringInfo.format(tamanhos[i], tempo[i], tempoPiorCaso[i], tempoMelhorCaso[i]))

    criar_Grafico(tamanhos, tempo, tempoPiorCaso, tempoMelhorCaso)

if __name__ == '__main__':

    main()