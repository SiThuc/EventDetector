
class Edge(object):
    def __init__(self, edgeId=None,fromNode=None, toNode=None, weight=None):
        self._mEdgeId = edgeId
        self._mFromNode = fromNode
        self._mToNode = toNode
        self._mWeight = weight

    @staticmethod
    def Edge_without_id( fromNode, toNode, weight):
        eg = Edge()
        eg._mFromNode = fromNode
        eg._mToNode = toNode
        eg._mWeight = weight
        return eg

    def getEdgeId(self):
        return self._mEdgeId

    def getFromNode(self):
        return self._mFromNode

    def getToNode(self):
        return self._mToNode

    def getWeight(self):
        return self._mWeight

    def __hash__(self):
        prime = 31
        result = 1
        if self._mFromNode == None:
            result = prime * result
        else:
            result = prime * result + self._mFromNode.__hash__()

        if self._mToNode == None:
            result = prime * result
        else:
            result = prime * result + self._mToNode.__hash__()
        return result

    def __eq__(self, other):
        if other == None:
            return False
        if not isinstance(other, type(self)):
            return False
        if self._mFromNode == None:
            if other._mFromNode !=None:
                return False
        elif not self._mFromNode == other._mFromNode:
            return False
        if self._mToNode == None:
            if other._mToNode !=None:
                return False
        elif not self._mToNode == other._mToNode:
            return False

        return True













