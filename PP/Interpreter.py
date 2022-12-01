class AbstractExpression():
    
    @staticmethod
    def interpret():
        """
        O método `interpreter` é chamado recursivamente para cada AbstractExpression
        """
class Number(AbstractExpression):
    "Terminal Expression"
    def __init__(self, value):
        self.value = int(value)
    def interpret(self):
        return self.value
    def __repr__(self):
        return str(self.value)
class Add(AbstractExpression):
    "Non-Terminal Expression."
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def interpret(self):
        return self.left.interpret() + self.right.interpret()
    def __repr__(self):
        return f"({self.left} Add {self.right})"
class Subtract(AbstractExpression):
    "Non-Terminal Expression"
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def interpret(self):
        return self.left.interpret() - self.right.interpret()
    def __repr__(self):
        return f"({self.left} Subtract {self.right})"
# The Client
# The sentence complies with a simple grammar of
# Number -> Operator -> Number -> etc,
SENTENCE = "5 + 4 - 3 + 7 - 2"
print(SENTENCE)
# Split the sentence into individual expressions that will be added 
# to an Abstract Syntax Tree (AST) as Terminal and Non-Terminal 
# expressions
TOKENS = SENTENCE.split(" ")
print(TOKENS)