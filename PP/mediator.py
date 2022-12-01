class Course(object):
    """classe Mediator"""
 
    def displayCourse(self, user, course_name):
        print("[curso de {}]: {}".format(user, course_name))
 
 
class User(object):
    '''Uma classe cujas instâncias desejam interagir umas com as outras.'''
 
    def __init__(self, name):
        self.name = name
        self.course = Course()
 
    def sendCourse(self, course_name):
        self.course.displayCourse(self, course_name)
 
    def __str__(self):
        return self.name
 
"""main"""
 
if __name__ == "__main__":
 
    mayank = User('Pedro')   # objeto usuario
    lakshya = User('Tiago') # objeto usuario
    krishna = User('João') # objeto usuario
 
    mayank.sendCourse("Análise e Desenvolvimento de sistemas")
    lakshya.sendCourse("Engenharia de Software")
    krishna.sendCourse("Ciência da computação")