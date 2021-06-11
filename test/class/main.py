# *args : tuple, **kwargs : dictionary
# args: arguments, kwargs: keyword arguments
# list : []
# tuple: () ex) (1,) : comma is necessary , missing () is allowed
# __call__  : making a class object callable
# __init__ : initializing a class object
class human:
    planet = "Earth"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __call__(self, *args, **kwargs):
        for key, value in kwargs.items():
            print("{0} is {1}".format(key, value))
        for value in args:
            print(value)

    def call(self, a, b, training=True):
        return a + b


class student(human):
    def __init__(self, univeresity, major, name, age):
        super(student, self).__init__(name, age)
        self.university = univeresity
        self.major = major

    def __call__(self, name):
        print('__call__')
        return name

    def call(self, a, b, training=False):
        return a * b


minsul = human("minsul", 27)

print(minsul.name)
print(minsul.age)
print(minsul.planet)

minsul_student = student('Kyunghee university', 'Mechanical engineering', 'Minsul Lee', 27)

print(minsul_student.name)
print(minsul_student.major)
print(minsul_student.planet)
print(minsul_student.university)
print(minsul_student.age)

print(callable(human))
a = minsul_student('minsulll')

print(minsul_student.call(10,10,training=True))
print(minsul.call(10,10,training=True))

#minsul(1, 2, 3, hahaha="hahaha~!")
