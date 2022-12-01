class State:

    def scan(self):
          
        """Passando o radio para a próxima estação"""
        self.pos += 1
  
        """verificando a última estação"""
        if self.pos == len(self.stations):
            self.pos = 0
        print("Visitando... estação {} {}".format(self.stations[self.pos], self.name))
  
"""Classe separada para o estado AM do rádio"""
class AmState(State):
  
    """construtor para classe de estado AM"""
    def __init__(self, radio):
          
        self.radio = radio
        self.stations = ["1250", "1380", "1510"]
        self.pos = 0
        self.name = "AM"
  
    """método para alternar o estado"""
    def toggle_amfm(self):
        print("Mudando para FM")
        self.radio.state = self.radio.fmstate
  
"""Classe separada para estado FM"""
class FmState(State):
  
    """Constriutor para estado FM"""
    def __init__(self, radio):
        self.radio = radio
        self.stations = ["81.3", "89.1", "103.9"]
        self.pos = 0
        self.name = "FM"
  
    """método para alternar o estado"""
    def toggle_amfm(self):
        print("Switching to AM")
        self.radio.state = self.radio.amstate
  
"""Classe do radio"""
class Radio:

    def __init__(self):
          
        """Temos um estado AM e um estado FM"""
        self.fmstate = FmState(self)
        self.amstate = AmState(self)
        self.state = self.fmstate
  
    """método para alternar o interruptor"""
    def toggle_amfm(self):
        self.state.toggle_amfm()
  
    """método para escanear"""
    def scan(self):
        self.state.scan()
  
""" main """
if __name__ == "__main__":
  
    """criando o objeto de rádio"""
    radio = Radio()
    actions = [radio.scan] * 3 + [radio.toggle_amfm] + [radio.scan] * 3
    actions *= 2
  
    for action in actions:
        action()
