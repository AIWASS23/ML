"""Classe memento para salvar os dados"""
 
class Memento:
 
    """Função construtora"""
    def __init__(self, file, content):
 
        """colocamos todo o conteúdo do seu arquivo aqui"""
         
        self.file = file
        self.content = content
 
"""Utilitário de gravação de arquivos"""
 
class FileWriterUtility:
 
    """Função construtora"""
 
    def __init__(self, file):
 
        """armazenando os dados do arquivo de entrada"""
        self.file = file
        self.content = ""
 
    """Gravando os dados no arquivo"""
 
    def write(self, string):
        self.content += string
 
    """salvando os dados no Memento"""
 
    def save(self):
        return Memento(self.file, self.content)
 
    """Recurso UNDO fornecido"""
 
    def undo(self, memento):
        self.file = memento.file
        self.content = memento.content
 
class FileWriterCaretaker:
 
    """salva os dados"""
 
    def save(self, writer):
        self.obj = writer.save()
 
    """desfazer o conteúdo"""
 
    def undo(self, writer):
        writer.undo(self.obj)
 
 
if __name__ == '__main__':
 
    """cria o objeto zelador"""
    caretaker = FileWriterCaretaker()
 
    """cria o objeto escritor"""
    writer = FileWriterUtility("GFG.txt")
 
    """escrevendo dados em arquivo usando o objeto escritor"""
    writer.write("Primeira frase escrita\n")
    print(writer.content + "\n\n")
 
    """Salvando o arquivo"""
    caretaker.save(writer)
 
    """novamente escrevendo usando o escritor"""
    writer.write("Segunda frase escrita\n")
 
    print(writer.content + "\n\n")
 
    """desfazendo o arquivo"""
    caretaker.undo(writer)
 
    print(writer.content + "\n\n")