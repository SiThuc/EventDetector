
class Array(object):
    def __init__(self, length):
        self._data = []
        for i in range(0,length):
            self._data.insert(i, 0)
    def update_value(self, index, value):
        self._data[index] += value

    def get_value(self, index):
        return self._data[index]

    def size(self):
        return len(self._data)

    def __str__(self):
        return str((self._data))

if __name__=="__main__":
    arr1 = Array(10)
    print("New array:", arr1)
    arr1.update_value(5, 2.5)
    print("Current array", arr1)
    print(arr1.get_value(9))
    print("Current length:", arr1.size())