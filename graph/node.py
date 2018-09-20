
class Node(object):
    def __init__(self, name, weight):
        self._name = name
        self._mWeight = weight
        self._mOutDegree = 0

    def getName(self):
        return self._name

    def getWeight(self):
        return self._mWeight

    def getOutDegree(self):
        return self._mOutDegree

    def updateOutDegree(self, degree):
        self._mOutDegree += degree
        
