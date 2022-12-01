import random

def gene_elem(nb_bit):
    L = []
    for i in range(nb_bit):
        L.append(random.randint(0, 1))
    return L

def gene_alea_list(nb_list,nb_bit): 
    L = []
    i = 0
    while(i < nb_list):
        l = []
        for j in range(nb_bit):
            l.append(random.randint(0, 1))
        if(l not in L):
            L.append(l)
            i += 1
    return L

def arbr_from_list(L):  
    AP = init_arbre(L[0])
    for i in range(1, len(L)):
        add(AP, L[i])
    return AP

def affiche(AP):
    for cle,val in AP.items():
        print(str(cle)+" : "+str(val))
    print("\n")

def init_arbre(L):  
    D = {}
    D[0] = [0,0,L,0,0,0,"N","N",0,[0]] 
    return D

def set_indice(AP): 
    L = AP[0][9]  
    if len(AP[0][9]) > 1:
        return AP[0][9].pop(-1)
    else:
        AP[0][9][0] = AP[0][9][0] + 1
        return AP[0][9][0]

def compar_list(L1,L2,ini): 
    N = len(L1)
    i = ini

    while(i < N and L1[i] == L2[i]  ):
        i += 1
    
    if i == N:
        return -2
    elif i == ini:
        return ini+1
    else:
        return i
    
def remonte_lien(AP, indice): 

    bitnum = AP[indice][1]
    current = indice
    List = AP[current][2]
    l = List[:]
    continu = True

    if List[bitnum] == 1: 
        l[bitnum] = 0  
        while current != 0 and continu:
            current = AP[current][5] 

            if l[0:bitnum] == AP[current][2][0:bitnum]:
                AP[indice][3] = current

                if AP[current][8] != current:
                    print(AP[current][8])
                    print("\n")
                AP[current][8]=indice
                continu=False

    elif List[bitnum] ==0 :
            l[bitnum] = 1

            while current != 0 and continu:
                current = AP[current][5]

                if l[0:bitnum] == AP[current][2][0:bitnum]:
                    AP[indice][4] = current

                    if AP[current][8] != current:
                        print(AP[current][8])
                        print("\n")
                    AP[current][8] = indice
                    continu=False

def add(AP,List):

    current = 0
    bitnum = 0
    new_indice = set_indice(AP)
    continu = True

    while(continu):

        L = AP[current][2]
        if List[AP[current][1]] == 0: 
            left = AP[current][3] 

            if ((AP[current][1]>=bitnum) and (bitnum < len(List)-1)):

                bitnum = compar_list(AP[current][2], List, bitnum) 
            if bitnum < 0:
                    return -1
            
            if AP[current][6] == "N": 
                AP[current][6] = "L"

                if AP[AP[current][3]][8] == current:
                    AP[AP[current][3]][8] = AP[current][3] 

                AP[current][3]=new_indice

                if bitnum >= len(List):
                    print("len(List) jogue à esquerda = "+str(len(List)))
                    AP[new_indice] = [new_indice, bitnum-1, List, new_indice, new_indice, current, "N", "N",new_indice]
                    continu = False

                elif bitnum < len(List):
                    AP[new_indice] = [new_indice, bitnum, List, new_indice, new_indice, current, "N", "N",new_indice]
                    remonte_lien(AP, new_indice)
                    continu = False
                else:
                    print("sem problema\n")
                    continu = False

            elif bitnum < AP[left][1]: 
                l = AP[left][2] 
                AP[current][3] = new_indice

                if l[bitnum] == 0:
                    AP[new_indice] = [new_indice, bitnum, List, left, new_indice, current,"L","N",new_indice]

                else:
                    AP[new_indice] = [new_indice, bitnum, List, new_indice, left, current,"N","R",new_indice]

                AP[left][5] = new_indice
                continu = False

            elif (bitnum >= AP[left][1] and AP[current][6] != "N"): 
                current=left

            else:
                print("\sem problemas no lado esquerdo \n")

        elif List[AP[current][1]] == 1:  
            right=AP[current][4] 

            if AP[current][1] >= bitnum:
                bitnum = compar_list(AP[current][2], List, bitnum)

            if bitnum < 0:
                    print("O elemento a inserir já está na árvore\n")
                    return -1

            if AP[current][7] == "N":
                AP[current][7] = "R"

                if AP[AP[current][4]][8] == current:
                    AP[AP[current][4]][8] = AP[current][4] 
                AP[current][4]=new_indice

                if bitnum >= len(List): 
                     AP[new_indice] = [new_indice, bitnum-1, List, new_indice, new_indice, current, "N", "N",new_indice]
                     continu = False

                elif bitnum < len(List):
                    AP[new_indice] = [new_indice, bitnum, List, new_indice, new_indice, current, "N", "N",new_indice]
                    remonte_lien(AP, new_indice)
                    continu = False

                else:
                    print("sem problemas no lado direito\n")
                    continu = False

            elif bitnum < AP[right][1]: 

                l=AP[right][2]  
                AP[current][4] = new_indice

                if l[bitnum] == 0:
                    AP[new_indice] = [new_indice, bitnum, List, right, new_indice, current,"L","N",new_indice]

                else:
                    AP[new_indice] = [new_indice, bitnum, List, new_indice, right, current,"N","R",new_indice]
                AP[right][5] = new_indice
                continu = False

            elif (bitnum>=AP[right][1] and AP[current][7] != "N"):
                current = right
            else:
                print("\Sem problemas \n")

    return new_indice
        
def find(AP, List): 

    current = 0
    bitnum = 0
    sizelist = len(List)
    continu = True

    while(continu):

        bitnum = AP[current][1] 
        if(List[bitnum] == 0): 

            left = AP[current][3] 

            if(AP[current][6] == "L"): 
                current = left

            elif(AP[current][6] == "N" and left == current): 

                if(AP[current][2][:] == List[:]): 
                    return current

                else:
                    return -1 

            elif(AP[current][6] == "N" and left != current): 

                if(AP[current][2][:] == List[:]): 
                    return current

                elif(AP[left][2][:] == List[:]):
                    return left

                else:
                    return -1 

        elif(List[bitnum] == 1): 
            right = AP[current][4]

            if(AP[current][7] == "R"):
                current = right

            elif(AP[current][7] == "N" and right == current):

                if(AP[current][2][:] == List[:]):
                    return current

                else:
                    return -1

            elif(AP[current][7] == "N" and right != current):

                if(AP[current][2][:] == List[:]):
                    return current

                elif(AP[right][2][:] == List[:]):
                    return right

                else:
                    return -1

def pop(AP,List):
    numpop = find(AP, List)

    if (numpop < 0):
        print("Erro: Este elemento não está em nossa árvore Patricia 🥲 \n")
        return -1

    else:
        parent = AP[numpop][5]
        indpar = AP[parent][1]
        res = AP[numpop][2]

        if AP[numpop][6] == "N" and AP[numpop][7] == "N":
            AP[AP[numpop][3]][8] = AP[numpop][3]
            AP[AP[numpop][4]][8] = AP[numpop][4]

            if AP[parent][3] == numpop:
                AP[parent][6] = "N"
                AP[parent][3] = parent

                if AP[parent][2][indpar] == 1:
                    remonte_lien(AP, parent)
               
            elif numpop == AP[parent][4]: 
                AP[parent][7] = "N"
                AP[parent][4] = parent

                if AP[parent][2][indpar] == 0: 
                    remonte_lien(AP, parent)

            AP[0][9].append(numpop)
            del AP[numpop]

        elif AP[numpop][6] == "L" and AP[numpop][7] == "N": 
            AP[AP[numpop][4]][8] = AP[numpop][4]

            if AP[parent][3] == numpop: 
                AP[parent][3]=AP[numpop][3]
                AP[AP[numpop][3]][5] = parent

            elif numpop == AP[parent][4]: 
                AP[parent][4] = AP[numpop][3]
                AP[AP[numpop][3]][5] = parent

            indpoi = AP[numpop][8]

            if indpoi != numpop:
                remonte_lien(AP, indpoi)

            AP[0][9].append(numpop)
            del AP[numpop]

        elif AP[numpop][6] == "N" and AP[numpop][7] == "R": 
            AP[AP[numpop][3]][8] = AP[numpop][3]

            if AP[parent][3] == numpop: 
                AP[parent][3] = AP[numpop][4]
                AP[AP[numpop][4]][5] = parent

            elif numpop == AP[parent][4]: 
                AP[parent][4] = AP[numpop][4]
                AP[AP[numpop][4]][5] = parent

            indpoi = AP[numpop][8]

            if indpoi != numpop:
                remonte_lien(AP, indpoi)

            AP[0][9].append(numpop)
            del AP[numpop] 

        elif AP[numpop][6] == "L" and AP[numpop][7] == "R":
            indpoi = AP[numpop][8] 
            parent_poi = AP[indpoi][5] 
            AP[numpop][8] = numpop
            ind_par_poi = AP[parent_poi][1] 

            if AP[parent_poi][3] == indpoi:
                AP[parent_poi][6] = "N"
                AP[parent_poi][3] = parent_poi

                if AP[parent_poi][2][ind_par_poi] == 1: 
                    remonte_lien(AP, parent_poi)
               
            elif indpoi == AP[parent_poi][4]: 
                AP[parent_poi][7] = "N"
                AP[parent_poi][4] = parent_poi

                if AP[parent_poi][2][ind_par_poi] == 0: 
                    remonte_lien(AP, parent_poi)

            AP[numpop][2] = AP[indpoi][2][:]
            AP[0][9].append(indpoi)
            del AP[indpoi]
    return res


def fonction_test(AP):

    N = len(AP) 
    n = len(AP[1][2]) 
    print(" Verificamos se os seguintes elementos estão contidos em nossa árvore Patricia ")
    print("- find(AP,"+str(AP[N//2][2])+"):")
    ind = find(AP, AP[N//2][2])

    if(ind == -1):
        print("Este elemento não está contido na Árvore Patrícia. 😒 \n")

    else:
        print("O seguinte elemento "+str(AP[N//2][2])+" pode ser encontrado no índice: "+str(ind)+" da lista de codificação para a árvore 😍 \n")

    l = gene_elem(n)
    print("- find(AP,"+str(l)+"):")
    ind = find(AP, l)

    if(ind == -1):
        print("Este elemento não está contido na Árvore Patrícia. 😒 \n")
    else:
        print("o seguinte elemento "+str(l)+" pode ser encontrado no índice: "+str(ind)+" da lista de codificação para a árvore 😍 \n")

    l = gene_elem(n)
    print("- find(AP,"+str(l)+"):")
    ind = find(AP, l)

    if(ind == -1):
        print("Este elemento não está contido na Árvore Patrícia. 😒 \n")
    else:
        print("o seguinte elemento "+str(l)+" pode ser encontrado no índice: "+str(ind)+" da lista de codificação para a árvore\n")

    print("print(AP):\n")
    affiche(AP)
    print("Vamos agora verificar a função pop():\n")
    print("- pop(AP,"+str(AP[N//2][2])+") :")
    l = pop(AP, AP[N//2][2])

    if l != -1:
        print("Elemento "+str(l)+" foi apagado da lista de codificação para a árvore Patricia 🥳.\n")

    print("- pop(AP,"+str(AP[N-2][2])+") :")
    l2 = []
    l2[:] = pop(AP, AP[N-2][2])

    if( l2 != -1):
        print("Elemento "+str(l2)+" foi apagado da lista de codificação para a árvore Patricia 🥳.\n")

    l = gene_elem(n)
    print("- pop(AP,"+str(l)+") :")

    if(pop(AP, l) != -1):
        print("Elemento "+str(l)+" foi apagado da lista de codificação para a árvore Patricia 🥳.\n")

    print("print(AP):\n")
    affiche(AP)
    print("Vamos agora verificar a função add():\n")
    print("- add(AP,"+str(l)+") :")
    ind = add(AP, l)

    if ind != -1:
        print("Elemento "+str(l)+" foi adicionado na árvore Patricia no índice: "+str(ind)+". 🥳 \n")
        
    print("- add(AP,"+str(l2)+") :")
    ind = add(AP ,l2)

    if ind != -1:
        print("Elemento "+str(l2)+" foi adicionado na árvore Patricia no índice: "+str(ind)+". 🥳 \n")

    print("Vamos imprimir a árvore Patrícia :\n")
    affiche(AP)


Lf = gene_alea_list(10, 8)
AP = arbr_from_list(Lf)
print("Esta é uma árvore Patrícia gerada aleatoriamente: \n ")
affiche(AP) 
fonction_test(AP)