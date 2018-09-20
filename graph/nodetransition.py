
class NodeTransition(object):
    def __init__(self, name, probability):
        self._name = name
        self._mTransitionProbability = probability

    def getNodeName(self):
        return self._name

    def getProbability(self):
        return self._mTransitionProbability

    def __str__(self):
        return "<" + str(self._name) + ", " + str(self._mTransitionProbability) + ">"