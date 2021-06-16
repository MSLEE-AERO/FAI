def minsul(name, *args, **kwargs):
    age = 'age'
    major = 'major'
    print(kwargs.keys())
    if (age and major) in kwargs.keys():
        age = kwargs.get(age)
        major = kwargs.get(major)
        print(age, major)
        raise ValueError('minsul')



for i in range(32):
    for j in range(182):
        for k in range(8):
            for l in range(8):
                print(i, j, k, l)

print('end')