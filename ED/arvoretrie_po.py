class Node:
    def __init__(self, key = None, data = None):
        self.key = key
        self.data = data
        self.children = dict()
    
    def addChild(self, key, data = None):
        if not isinstance(key, Node):
            self.children[key] = Node(key, data)
        else:
            self.children[key.label] = key

class Trie:
    def __init__(self):
        self.head = Node()
    
    def add_word(self, word):
        current_node = self.head
        word_finished = True
        
        for i in range(len(word)):
            if word[i] in current_node.children:
                current_node = current_node.children[word[i]]
            else:
                word_finished = False
                break
        
        if not word_finished:
            while i < len(word):
                current_node.addChild(word[i])
                current_node = current_node.children[word[i]]
                i += 1
        
        current_node.data = word
    
    def add_words(self, words):
        for word in words.split():
            self.add_word(word)
   
    def has_word(self, word):
        if word == '':
            return False
        if word == None:
            raise ValueError('Trie.has_word precisa de uma string valida.')

        current_node = self.head
        exists = True
        
        for letter in word:
            if letter in current_node.children:
                current_node = current_node.children[letter]
            else:
                exists = False
                break
        
        if exists:
            if current_node.data == None:
                exists = False
        
        return exists

    def remove_word(self, word):
        if word == '':
            return False
        if word == None:
            raise ValueError('Trie.has_word precisa de uma string valida')

        current_node = self.head
        exists = True
        
        for letter in word:
            if letter in current_node.children:
                current_node = current_node.children[letter]
            else:
                exists = False
                break
        
        if exists:
            current_node.data = None


if __name__ == '__main__':

    trie = Trie()
    words = 'sol sal sola cafe pao padeiro padaria'

    trie.add_words(words)

    #Testes para busca
    print("Teste para busca:\n")
    if(trie.has_word('sol') == True):
        print("A palavra foi encontrada.\n")
    else:
        print("A palavra nao foi encontrada.\n")      

    if(trie.has_word('cobre') == True):
        print("A palavra foi encontrada.\n")
    else:
        print("A palavra nao foi encontrada.\n")

    #Testes para inserção
    print("Teste para insercao:\n")
    if(trie.has_word('so') == True):
        print("A palavra foi encontrada.\n")
    else:
        print("A palavra nao foi encontrada.\n")  
    
    trie.add_word('so')

    if(trie.has_word('so') == True):
        print("A palavra foi encontrada.\n")
    else:
        print("A palavra nao foi encontrada.\n")  

    #Testes para remoção
    print("Teste para remocao:\n")
    trie.remove_word('sol')
    
    if(trie.has_word('sol') == True):
        print("A palavra foi encontrada.\n")
    else:
        print("A palavra nao foi encontrada.\n")