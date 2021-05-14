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


class student(human):
    def __init__(self, univeresity, major, name, age):
        super().__init__(name, age)
        self.university = univeresity
        self.major = major


minsul = human("minsul",27)

print(minsul.name)
print(minsul.age)
print(minsul.planet)

minsul_student = student('Kyunghee university','Mechanical engineering','Minsul Lee',27)

print(minsul_student.name)
print(minsul_student.major)
print(minsul_student.planet)
print(minsul_student.university)
print(minsul_student.age)


print(callable(human))

minsul(1,2,3,hahaha="hahaha~!")



