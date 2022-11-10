import numpy as np
import time
from matplotlib import pyplot as plt
from numpy import random


def criar_Grafico(
    x,
    ym,
    z,
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('shell')
    ax.set_ylabel('Tempo')
    ax.set_xlabel('Tamanho da Lista')
    ax.plot(x, ym, label='Caso Aleatório')
    ax.plot(x, z, label='Pior Caso')

    def calculaPasso(fim, ini=0):
        i = fim - ini
        j = 0
        while i > 10:
            j += 1
            i /= 10
        i = int(i)
        if i >= 5:
            return 10**(j)
        else:
            return 5 * (10**(j - 1))

    y = max(ym[-1], z[-1])
    passoX = calculaPasso(x[-1], x[0])
    passoY = calculaPasso(y)

    plt.xticks(np.arange(x[0], x[-1] + passoX, step=passoX))
    plt.yticks(np.arange(0, y + passoY, step=passoY))
    ax.legend()

    fig.savefig('graph.png')


def shellSort(array, piorCaso=False):
    n = len(array)
    if piorCaso:
        h = 2.2
    else:
        h = 1
        while h < n:
            h = 3 * h + 1
    while h > 0:
        if piorCaso:
            h = int(h / 2.2)
        else:
            h = (h - 1) // 3
        for i in range(h, n):
            c = array[i]
            j = i
            while j >= h and c < array[j - h]:
                array[j] = array[j - h]
                j = j - h
            array[j] = c


def listaAleatoria(tamanho, nivel=1):
    lista = []
    maximo = tamanho * (10**nivel)
    for i in range(tamanho):
        lista.append(random.randint(maximo))
    return lista


def main():
    tamanhos = [10000, 20000, 30000, 40000, 50000, 80000, 90000]
    tempo = []
    tempoPiorCaso = []

    for i in range(len(tamanhos)):
        lista = listaAleatoria(tamanhos[i])
        lista2 = lista.copy()

        start = time.time()
        shellSort(lista)
        end = time.time()
        tempo.append(end - start)

        start = time.time()
        shellSort(lista2, piorCaso=True)
        end = time.time()
        tempoPiorCaso.append(end - start)

        stringInfo = 'Lista de tamanho {} - Caso aleatório: {:.4f}s; Pior Caso: {:.4f}s'
        print(stringInfo.format(tamanhos[i], tempo[i], tempoPiorCaso[i]))

    criar_Grafico(tamanhos, tempo, tempoPiorCaso)


if __name__ == '__main__':
    main()