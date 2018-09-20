import heapq
from utils.array import Array
class Propagator(object):
    def __init__(self, graph):
        self._mGraph = graph
        self._mGraphSize = graph.getNodeCnt()
        self._mapFromIdToStrng = dict()
        self._mapFromStringToId = dict()
        #self._queue = []
        #self._mScore = []
        #self._propagationScore = []
        for i in range(0, self._mGraph.getNodeCnt()):
            self._mapFromIdToStrng[i] = graph._mNodes[i].getName()
            self._mapFromStringToId[graph._mNodes[i].getName()] = i


    def search(self, queryNode, epsilon, c, errorBound):
        idxQueryId = self._mapFromStringToId[queryNode]
        self.initialize(idxQueryId, c)
        iter = 0
        while len(self._queue) > 0 and self._queue[0][0] > c * errorBound:
            # nodeId = heapq.heappop(self._queue)
            val, nodeId = heapq.heappop(self._queue)
            self.propagate(nodeId, c)
            self._propagationScore.update_value(nodeId, 0.0)
            iter += 1
            if iter >= 2000:
                break
        results = dict()
        for i in range(0, self._mScore.size()):
            if self._mScore.get_value(i) >= epsilon:
                results[self._mapFromIdToStrng[i]] = self._mScore.get_value(i)
        return results

    def initialize(self, queryId, c):
        self._mScore = Array(self._mGraphSize)
        self._propagationScore = Array(self._mGraphSize)

        self._mScore.update_value(queryId, c)
        self._propagationScore.update_value(queryId, c)

        self._queue = []
        heapq.heappush(self._queue,(self._propagationScore.get_value(queryId), queryId))

    def check_in_queue(self, neighborId):
        check = False
        for ele in self._queue:
            if ele[1] == neighborId:
                return True
        return check

    def remove_from_queue(self, neighborId):
        for ele in self._queue:
            if ele[1] == neighborId:
                self._queue.remove(ele)
                break

    def update_mScore(self,index, delta):
        if len(self._mScore) <= index:
            self._mScore.insert(index, delta)
        else:
            self._mScore[index] += delta

    def update_propagationScore(self, index, delta):
        if len(self._propagationScore) <= index:
            self._propagationScore.insert(index, delta)
        else:
            self._propagationScore[index] += delta

    # propagate te delta rwr from node id to its in-neighbors
    def propagate(self, nodeId, c):
        toPropagteScore = self._propagationScore.get_value(nodeId)
        nameOfNodeId = self._mapFromIdToStrng[nodeId]
        inTransitions = self._mGraph.getIncomingTransitions(nameOfNodeId)

        for nt in inTransitions:
            neighorNode = nt.getNodeName()
            #compute the delta core for the neighbor
            probability = nt.getProbability()
            deltaScore = (1-c)* probability * toPropagteScore
            neighborId = self._mapFromStringToId[neighorNode]
            # print("NeighborID:",neighborId)
            # print("mScore:")
            # for ele in self._mScore:
            #     print(ele)
            self._mScore.update_value(nodeId, deltaScore)
            #self._mScore[neighborId] += deltaScore

            #update the propagation score for the neighbor
            self._propagationScore.update_value(neighborId, deltaScore)
            #self._propagationScore[neighborId] += deltaScore

            #update the heap
            if self.check_in_queue(neighborId):
                self.remove_from_queue(neighborId)
            heapq.heappush(self._queue,(self._propagationScore.get_value(neighborId), neighborId))