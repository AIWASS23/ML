class Subject:
 
    """Representa o que está sendo observado"""
 
    def __init__(self):
 
        """cria uma lista de observadores vazia"""
 
        self._observers = []
 
    def notify(self, modifier = None):
 
        """Alerta os observadores"""
 
        for observer in self._observers:
            if modifier != observer:
                observer.update(self)
 
    def attach(self, observer):
 
        """Se o observador não estiver na lista, acrescente-o à lista"""
 
        if observer not in self._observers:
            self._observers.append(observer)
 
    def detach(self, observer):
 
        """Remove o observador da lista de observadores"""
 
        try:
            self._observers.remove(observer)
        except ValueError:
            pass
 
 
 
class Data(Subject):
 
    """monitorando o objeto"""
 
    def __init__(self, name =''):
        Subject.__init__(self)
        self.name = name
        self._data = 0
 
    @property
    def data(self):
        return self._data
 
    @data.setter
    def data(self, value):
        self._data = value
        self.notify()
 
 
class HexViewer:
 
    """atualiza o Hewviewer"""
 
    def update(self, subject):
        print('HexViewer: Subject {} has data 0x{:x}'.format(subject.name, subject.data))
 
class OctalViewer:
 
    """atualiza o OctalViewer"""
 
    def update(self, subject):
        print('OctalViewer: Subject' + str(subject.name) + 'has data '+str(oct(subject.data)))
 
 
class DecimalViewer:
 
    """atualiza o Decimal viewer"""
 
    def update(self, subject):
        print('DecimalViewer: Subject % s has data % d' % (subject.name, subject.data))
 
"""main"""
 
if __name__ == "__main__":
 
    """fornece os dados"""
 
    obj1 = Data('Data 1')
    obj2 = Data('Data 2')
 
    view1 = DecimalViewer()
    view2 = HexViewer()
    view3 = OctalViewer()
 
    obj1.attach(view1)
    obj1.attach(view2)
    obj1.attach(view3)
 
    obj2.attach(view1)
    obj2.attach(view2)
    obj2.attach(view3)
 
    obj1.data = 10
    obj2.data = 15