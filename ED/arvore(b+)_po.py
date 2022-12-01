class Node(object):
    
    def __init__(self, order):
        self.order = order
        self.keys = []
        self.values = []
        self.leaf = True

    def add(self, key, value):

        # Se o nó estiver vazio, basta inserir o par de valores-chave.
        if not self.keys:
            self.keys.append(key)
            self.values.append([value])
            return None

        for i, item in enumerate(self.keys):
            # Se a nova chave corresponder à chave existente, adicione à lista de valores.
            if key == item:
                self.values[i].append(value)
                break

            # Se a nova chave for menor do que a chave existente, insira a nova chave à esquerda da chave existente.
            elif key < item:
                self.keys = self.keys[:i] + [key] + self.keys[i:]
                self.values = self.values[:i] + [[value]] + self.values[i:]
                break

            # Se a nova chave for maior do que todas as chaves existentes, insira a nova chave à direita de todas
            # chaves existentes.
            elif i + 1 == len(self.keys):
                self.keys.append(key)
                self.values.append([value])

    def split(self):
        left = Node(self.order)
        right = Node(self.order)
        mid = self.order // 2

        left.keys = self.keys[:mid]
        left.values = self.values[:mid]

        right.keys = self.keys[mid:]
        right.values = self.values[mid:]

        # Quando o nó é dividido, defina a chave pai para a chave mais à esquerda do nó filho direito.
        self.keys = [right.keys[0]]
        self.values = [left, right]
        self.leaf = False

    def is_full(self):
        return len(self.keys) == self.order

    def show(self, counter=0):
        print(counter, str(self.keys))

        # Imprime recursivamente a chave dos nós filhos (se houver).
        if not self.leaf:
            for item in self.values:
                item.show(counter + 1)

class BPlusTree(object):
    
    def __init__(self, order=8):
        self.root = Node(order)

    def _find(self, node, key):
        for i, item in enumerate(node.keys):
            if key < item:
                return node.values[i], i

        return node.values[i + 1], i + 1

    def _merge(self, parent, child, index):
        parent.values.pop(index)
        pivot = child.keys[0]

        for i, item in enumerate(parent.keys):
            if pivot < item:
                parent.keys = parent.keys[:i] + [pivot] + parent.keys[i:]
                parent.values = parent.values[:i] + child.values + parent.values[i:]
                break

            elif i + 1 == len(parent.keys):
                parent.keys += [pivot]
                parent.values += child.values
                break

    def insert(self, key, value):
        parent = None
        child = self.root

        # Percorra a árvore até que o nó folha seja alcançado.
        while not child.leaf:
            parent = child
            child, index = self._find(child, key)

        child.add(key, value)

        # Se o nó folha estiver cheio, divida o nó folha em dois.
        if child.is_full():
            child.split()

            # Depois que um nó folha é dividido, ele consiste em um nó interno e dois nós folha. Estes
            # precisa ser reinserido na árvore.
            if parent and not parent.is_full():
                self._merge(parent, child, index)

    def retrieve(self, key):
        child = self.root

        while not child.leaf:
            child, index = self._find(child, key)

        for i, item in enumerate(child.keys):
            if key == item:
                return child.values[i]

        return None

    def show(self):
        self.root.show()

def demo_node():
    print('Initializing node...')
    node = Node(order=4)

    print('\nInserting key a...')
    node.add('a', 'alpha')
    print('Is node full?', node.is_full())
    node.show()

    print('\nInserting keys b, c, d...')
    node.add('b', 'bravo')
    node.add('c', 'charlie')
    node.add('d', 'delta')
    print('Is node full?', node.is_full())
    node.show()

    print('\nSplitting node...')
    node.split()
    node.show()

def demo_bplustree():
    print('Initializing B+ tree...')
    bplustree = BPlusTree(order=4)

    print('\nB+ tree with 1 item...')
    bplustree.insert('a', 'alpha')
    bplustree.show()

    print('\nB+ tree with 2 items...')
    bplustree.insert('b', 'bravo')
    bplustree.show()

    print('\nB+ tree with 3 items...')
    bplustree.insert('c', 'charlie')
    bplustree.show()

    print('\nB+ tree with 4 items...')
    bplustree.insert('d', 'delta')
    bplustree.show()

    print('\nB+ tree with 5 items...')
    bplustree.insert('e', 'echo')
    bplustree.show()

    print('\nB+ tree with 6 items...')
    bplustree.insert('f', 'foxtrot')
    bplustree.show()

    print('\nRetrieving values with key e...')
    print(bplustree.retrieve('e'))

if __name__ == '__main__':
    demo_node()
    print('\n')
    demo_bplustree()