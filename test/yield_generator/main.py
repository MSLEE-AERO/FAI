def minsul(data):
    yield data[0] + data[1]

def minsul1(data):
    return data[0] + data[1]
data = [1,2]
if (__name__ == "__main__"):
    print(minsul(data))
    print(minsul1(data))
    for i in minsul(data):
        print(i)




