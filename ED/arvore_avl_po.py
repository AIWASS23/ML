class TreeNode(object): 
    def __init__(self, value): 
        self.value = value 
        self.left = None
        self.right = None
        self.height = 1
        
class AVL_Tree(object): 
   
    def insert(self, root, value): 
               
        #Implementando a arvore:
        if not root: 
            return TreeNode(value) 
        elif value < root.value: 
            root.left = self.insert(root.left, value) 
        else: 
            root.right = self.insert(root.right, value)
       
        #Mundando a altura      
        root.height = 1 + max(self.Height(root.left), self.Height(root.right)) 

        #Encontrando o fator de equilibrio
        balance = self.check_Avl(root) 

        #Verificando e usando as rotaçoes para balancear a arvore
        
        #Rotação para a Direita
        if balance > 1 and value < root.left.value: 
            return self.RR(root) 

        #Rotação para a esquerda
        if balance < -1 and value > root.right.value: 
            return self.LL(root) 
        #Rotação da árvore para a esquerda e depois para a direita
        if balance > 1 and value > root.left.value: 
            root.left = self.LL(root.left) 
            return self.RR(root) 
        #Rotação da árvore para a direita e depois para a esquerda
        if balance < -1 and value < root.right.value: 
            root.right = self.RR(root.right) 
            return self.LL(root) 
  
        return root 
     #Rotações
    def LL(self, node): 
       
        p = node.right 
        t = p.left

        #Rotações:
        p.left = node 
        node.right = t 

        #Mundando a altura: 
        node.height = 1 + max(self.Height(node.left), self.Height(node.right)) 
        p.height = 1 + max(self.Height(p.left), self.Height(p.right)) 
   
        return p 

    #Rotações
    def RR(self, node): 
  
        p = node.left 
        t = p.right

        #Rotações:
        p.right = node
        node.left = t 

        #Mundando a altura:
        node.height = 1 + max(self.Height(node.left), self.Height(node.right)) 
        p.height = 1 + max(self.Height(p.left), self.Height(p.right)) 
        return p 

    #Pegando a altura:
    def Height(self, root): 
        if not root: 
            return 0
  
        return root.height 

    #Pegando o Fator de Equilíbrio
    def check_Avl(self, root): 
        if not root: 
            return 0
  
        return self.Height(root.left) - self.Height(root.right) 
  
    def preOrder(self, root): 
  
        if not root: 
            return
  
        print("{0} ".format(root.value), end="") 
        self.preOrder(root.left) 
        self.preOrder(root.right) 


def insert_data(_data):
        mytree = AVL_Tree()
        root = None
        for i in _data:
            root = mytree.insert(root,i)
        print("A ordem da árvore AVL construída é:")
        mytree.preOrder(root)
        print()

if __name__ == "__main__":
    insert_data([1,2,3,4,5])
else:
    pass