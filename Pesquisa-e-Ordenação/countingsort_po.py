from random import randint
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

def countingSort(array):
    
    counter = [0] * ( max(array) + 1 )
    for i in array:
        counter[i] += 1

    ndx = 0;
    for i in range(len(counter)):
        while 0 < counter[i]:
            array[ndx] = i
            ndx += 1
            counter[i] -= 1
    return array

def criar_Grafico(x, y, z, k):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Counting Sort')
    ax.set_ylabel('Tempo')
    ax.set_xlabel('Tamanho da Lista')
    ax.plot(x, y, label='Caso Aleatório')
    ax.plot(x, z, label='Em ordem Crescente')
    ax.plot(x, k, label ='Em ordem Decrescente')
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
    emOrdemCrescente = []
    emOrdemDecrescente = []

    for i in range(len(tamanhos)):

        lista = gerarLista(tamanhos[i])
        
        start = time.time()
        countingSort(lista)
        end = time.time()
        tempo.append(end - start)

        start = time.time()
        countingSort(lista)
        end = time.time()
        emOrdemCrescente.append(end - start)

        lista.reverse()

        start = time.time()
        countingSort(lista)
        end = time.time()
        emOrdemDecrescente.append(end - start)

        stringInfo = 'Lista de tamanho {} - Caso aleatório: {:.4f}s; Em ordem crescente: {:.4f}s; Em ordem decrescente: {:.4f}s;'
        print(stringInfo.format(tamanhos[i], tempo[i], emOrdemCrescente[i], emOrdemDecrescente[i]))

    criar_Grafico(tamanhos, tempo, emOrdemCrescente, emOrdemDecrescente)

if __name__ == '__main__':
    main()