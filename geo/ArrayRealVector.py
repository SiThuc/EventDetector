import numpy as np
from scipy.spatial import distance
from copy import deepcopy

class ArrayRealVector():
    def __init__(self, lng=0.0, lat=0.0):
        self._data = []
        self._data.insert(0,lng)
        self._data.insert(1,lat)

    @staticmethod
    def ArrayRealVector_with_Dim(d):
        vt = ArrayRealVector()
        return vt

    def getDimension(self):
        return(len(self._data))

    def mapMultiply(self, d):
        n = ArrayRealVector()
        n._data[0] = self._data[0] *d
        n._data[1] = self._data[1] *d
        return n

    def mapDivideToSelf(self, d):
        self._data[0] = self._data[0]/d
        self._data[1] = self._data[1]/d

    def add(self, other):
        vt = ArrayRealVector()
        vt._data = np.add(self._data, other._data)
        return vt

    def ebeMultiply(self, other):
        vt = ArrayRealVector()
        vt._data = np.multiply(self._data, other._data)
        return vt

    def mapDivide(self, d):
        vt = ArrayRealVector()
        vt._data[0] = self._data[0]/ d
        vt._data[1] = self._data[1] / d
        return vt

    def subtract(self, other):
        vt = ArrayRealVector()
        vt._data[0] = self._data[0] - other._data[0]
        vt._data[1] = self._data[1] - other._data[1]
        return vt

    def getDistance(self, other):
        return distance.euclidean(self._data, other._data)

    def copy(self):
        return deepcopy(self)

    def getL1Norm(self):
        return np.linalg.norm(self._data, ord=1)

    def getNorm(self):
        return float(np.linalg.norm(self._data, ord=2))

    def __eq__(self, other):
        if not isinstance(other, type(self)):return NotImplemented
        vt1 = np.array(self._data)
        vt2 = np.array(other._data)
        return (vt1==vt2).all()

    def __hash__(self):
        return hash((self._data[0],self._data[1]))



if __name__ == "__main__":
    vt1 = ArrayRealVector(1, 4)
    print(vt1._data)
    vt2 = ArrayRealVector(3, 4)
    print(vt2._data)
    vt3 = ArrayRealVector.ArrayRealVector_with_Dim(2)
    print(vt3._data)
    print("dimension:", vt1.getDimension())
    print("mapMultiply vt1 * d:", vt1.mapMultiply(3)._data)
    vt1.mapDivideToSelf(3)
    print("mapDivideToSelf:", vt1._data)
    print("add():", vt1.add(vt2)._data)
    print("ebeMultiply:", vt1.ebeMultiply(vt2)._data)
    print("mapDivide:", vt2.mapDivide(3)._data)
    vt4 = ArrayRealVector(2,3)
    vt5 = ArrayRealVector(1,6)
    print("vt4", vt4._data)
    print("vt5", vt5._data)
    print("subtract",vt4.subtract(vt5)._data)
    print("Distance:", vt4.getDistance(vt5))
    vt6 = vt5.copy()
    print("copy vt6 from vt5",vt6._data)
    print("vt5 equals vt6:",vt5.__eq__(vt6))
    print("vt5 equals vt4:", vt5.__eq__(vt4))

    print("L1 Norm of vt4: %d"%vt4.getL1Norm())
    print("L2 Norm of vt4: %f"%vt4.getNorm())
    # vt3 = vt1.add(vt2)
    # print(type(vt3))
    # print(vt3._vector)
    # vt2 = ArrayRealVector(3,7)
    # print(vt2._vector)
    # print("distance: %f"%vt1.getDistance(vt2))
    # vt3 = vt1.copy()
    # print(vt3._vector)
    # vt3._vector[0] = 6
    # print("-----------")
    # print(vt1._vector)
    # print(vt3._vector)



