import time
import pickle
from graph.nodetransition import NodeTransition
from graph.progagator import Propagator
from graph.node import Node
from graph.edge import Edge

class Graph(object):
    def __init__(self):
        self._mNodes = []
        self._mEdges = []
        # keep the vicinity for each node.
        self._vicinity = dict()
        self._mOutEdges = dict()          # the outgoing edges of nodes
        self._mInEdges = dict()           #the incoming edges of nodes
        # the row - normalized transition matrix
        self._mTransition = dict()
        # the transpose of the row - normalized transition matrix
        self._mTransitionTranspose = dict()

        # stats
        self._timeCalcVicinity = None # time for computing vicinity.
        self._timeCreateGraph = None  # time for creating graph from begining.
        # Edges to write to edgeFile
        self._writtenEdges = []
        # Map from EntityId to NodeId
        self._mapIdtoId = dict()
        # Map buffEntityNode for creating edge
        self._buffEntityNode = dict()
        self._containedNode = set()

    def addEdge(self, edge):
        if not edge in self._mEdges:
            self._mEdges.append(edge)
            # update the incoming and outdoing neighbors of nodes
            edgeId = edge.getEdgeId()
            self._mOutEdges[edge.getFromNode()].add(edgeId)
            self._mInEdges[edge.getToNode()].add(edgeId)
            # update the degree information of nodes
            index = -1
            for i in range(0, len(self._mNodes)):
                if self._mNodes[i].getName() == edge.getFromNode():
                    index = i
                    break
            self._mNodes[index].updateOutDegree(edge.getWeight())
            return True
        else:
            # update the degree information of nodes
            index = -1
            for i in range(0, len(self._mNodes)):
                if self._mNodes[i].getName() == edge.getFromNode():
                    index = i
                    break
            self._mNodes[index].updateOutDegree(edge.getWeight())
            return False

    def getNode(self, nodeId):
        return self._mNodes[nodeId]

    def getEdge(self, edgeId):
        return self._mEdges[edgeId]

    def getNodeCnt(self):
        return len(self._mNodes)

    def getEdgeCnt(self):
        return len(self._mEdges)

    def getOutEdges(self, nodeName):
        return self._mOutEdges[nodeName]

    def getInEdges(self, nodeId):
        return self._mInEdges[nodeId]

    def getOutDegree(self, nodeID):
        degree = 0
        edgeIds = self._mOutEdges[nodeID]
        for edgeId in edgeIds:
            degree += self._mEdges[edgeId].getWeight()
        return degree

    # get the outgoing neighbors of a given node
    def getOutNeighbors(self, nodeName):
        outNeighbors = []
        # the set of outgoing edge ids
        edgeSet = self._mOutEdges[nodeName]
        for edgeId in edgeSet:
            outNeighbors.append(self.getEdge(edgeId).getToNode())
        return outNeighbors

    # get the incoming neighbors of a given node
    def getInNeighbors(self, nodeName):
        inNeighbors = []
        # the set of outgoing edge ids
        edgeSet = self._mInEdges[nodeName]
        for edgeId in edgeSet:
            inNeighbors.append(self.getEdge(edgeId).getFromNode())
        return inNeighbors

    def getOutgoingTransitions(self, nodeId):
        return self._mTransition[nodeId]

    def getIncomingTransitions(self, nodeName):
        return self._mTransitionTranspose[nodeName]

    # construct the row-nomalized transition matrix
    def constructTransitionMatrix(self):
        for node in self._mNodes:
            nodeName = node.getName()
            outDegree = node.getOutDegree()
            transitionList = []

            outEdges = self.getOutEdges(nodeName)
            for edgeId in outEdges:
                toNode = self.getEdge(edgeId).getToNode()
                pTransition = float(self.getEdge(edgeId).getWeight())/float(outDegree)
                transitionList.append(NodeTransition(toNode, pTransition))
            self._mTransition[nodeName] = transitionList

    # construct the transpose of the row-normalized transition matrix
    # it answers: given a node, which other node can directly walk to it, and with what probabilities?
    def constructTransitionTransposeMatrix(self):
        for i in range(0, self.getNodeCnt()):
            transitionList = []
            self._mTransitionTranspose[self._mNodes[i].getName()] = transitionList

        for node in self._mNodes:
            fromNode = node.getName()
            outDegree = node.getOutDegree()
            outEdges = self.getOutEdges(fromNode)
            for edgeId in outEdges:
                toNode = self.getEdge(edgeId).getToNode()
                pTransition = float(self.getEdge(edgeId).getWeight())/float(outDegree)
                transitionList = self._mTransitionTranspose[toNode]
                transitionList.append(NodeTransition(fromNode, pTransition))

    # compute the rwr scores for all the nodes.
    # epsilon: the rwr threshold; error bound: the rwr error; c: the restart probability
    def calcVicinity(self, epsilon, errorBound, c):
        searcher = Propagator(self)
        cnt = 0
        start = time.time()
        for n in self._mNodes:
            neighbors = searcher.search(n.getName(), epsilon, c, errorBound)
            self._vicinity[n.getName()] = neighbors
            cnt += 1
            if cnt % 100 == 0:
                print("Finished computing vicinity for %d nodes."%cnt)
        end = time.time()
        self._timeCalcVicinity = end - start

    def getRWR(self, fromWord, toWord):
        if not(toWord in self._vicinity.keys()):
            return 0.0
        else:
            neighbors = self._vicinity[toWord]

        if fromWord in neighbors:
            return neighbors[fromWord]
        else:
            return 0.0

    def getVicinity(self, nodeName):
        return self._vicinity[nodeName]


    def setVicinity(self, vicinity):
        self._vicinity = vicinity

    def numNode(self):
        return len(self._mNodes)

    def numEdges(self):
        return len(self._mEdges)

    def printStats(self):
        s = "Graph Stats:"
        s += " numNode=" + self.numNode()
        s += "; numEdges=" + self.numEdges()
        s += "; timeCalcVicinity=" + self._timeCalcVicinity
        print(s)

    def getTimeCalcVicinity(self):
        return self._timeCalcVicinity

    def generateNodes(self, inQueriedTweets):
        tweets = inQueriedTweets
        for tweet in tweets:
            entities = tweet.getEntities()
            #Nodes establishment
            for k in range(0, len(entities)):
                if not entities[k] in self._containedNode:
                    nNode = Node(entities[k], 1)
                    self._mNodes.append(nNode)
                    self._containedNode.add(entities[k])
                    self._buffEntityNode[entities[k]] = nNode
        print("Creating graph's node completed. Number of nodes:%d"%self.getNodeCnt())

        # initialize empty neighbor sets for each node
        for i in range(0, len(self._mNodes)):
            outEdges = set()
            inEdges = set()
            self._mOutEdges[self._mNodes[i].getName()] = outEdges
            self._mInEdges[self._mNodes[i].getName()] = inEdges

    def generateEdges(self, inQueriedTweets, directed):
        edgeIDcount = 0
        tweets = inQueriedTweets
        tempEdges = dict()
        for tweet in tweets:
            entities = tweet.getEntities()
            # Edges establishment
            for k in range(0, len(entities)-1):
                for j in range(k+1, len(entities)):
                    if entities[k] == entities[j]:
                        continue
                    fromNode = entities[k]
                    toNode = entities[j]
                    nEdge = Edge.Edge_without_id(fromNode, toNode, 1)
                    hCode = nEdge.__hash__()
                    if hCode in tempEdges:             #   tempEdges.__contains__(hCode)):
                        tempEdges[hCode]._mWeight += 1
                    else:
                        tempEdges[hCode] = nEdge
        print("There are %d edges in buff"%len(tempEdges))
        # Load edges into mEdges
        col = tempEdges.values()
        for edge in col:
            officialEdge = Edge(edgeIDcount, edge.getFromNode(), edge.getToNode(), edge.getWeight())
            check = self.addEdge(officialEdge)
            if check :
                edgeIDcount += 1
            if directed == False:
                reservedEdge = Edge(edgeIDcount, edge.getToNode(), edge.getFromNode(), edge.getWeight())
                check2 = self.addEdge(reservedEdge)
                if check2:
                    edgeIDcount+=1
        print("Loading edges completed. Number of edges:%d"%len(self._mEdges))
        self.constructTransitionMatrix()
        self.constructTransitionTransposeMatrix()
        print("Constructing transition matrices completed!")

    def getCreateTime(self):
        return self._timeCreateGraph

    def setCreateTime(self, duration):
        self._timeCreateGraph = duration

    def loadNodes(self, nodeFile):
        with open(nodeFile, "rb") as input:
            self._mNodes = pickle.load(input)
    def loadEdges(self, edgeFile):
        with open(edgeFile, "rb") as input:
            self._mEdges = pickle.load(input)

    def loadVicinity(self, vicinityFile):
        with open(vicinityFile, "rb") as input:
            self._vicinity = pickle.load(input)