import os
import re
import time
import errno
import pickle
from config import Config

from geo.geotweet import GeoTweet
from geo.tweetdatabase import TweetDatabase
from graph.graph import Graph
from pathlib import Path
from graph.progagator import Propagator

class Database():

    def __init__(self, config):
        self._config = config
        self._reader = None
        self._initialTweets = TweetDatabase()
        self._graph = None

    def getEntityGraph(self):
        return self._graph

    def getInitialTweets(self):
        return self._initialTweets

    def loadInitialTweets(self, tweetFile, numInitTweets):
        self._initialTweets = TweetDatabase()
        self._reader = open(tweetFile)

        for i in range(0, numInitTweets):
            line = self._reader.readline()
            tweet = GeoTweet.geoTweetFromAString(tweetString=line)
            self._initialTweets.add(tweet)
        print("There are %d tweets are loaded as initial tweets."%len(self.getInitialTweets().getTweets()))

    def nextTweet(self):
        tweet = None
        while tweet is None or tweet.numEntity() == 0:
            line = self._reader.readline()
            if line == '':
                return None
            tweet = GeoTweet.geoTweetFromAString(line)
        return tweet

    def generateEntityGraph(self, tdb, epsilon, errorBound, pRestart):
        start = time.time()
        # 1. init Graph
        self._graph = Graph()
        self._graph.generateNodes(tdb.getTweets())
        self._graph.generateEdges(tdb.getTweets(), False)
        self._graph.calcVicinity(epsilon, errorBound, pRestart)
        end = time.time()
        duration = end - start
        self._graph.setCreateTime(duration)

        bGraphTime = tdb.getStartTimestamp()
        eGraphTime = tdb.getEndTimestamp()

        self.createFolder(bGraphTime, eGraphTime)
        self.writeNode(self._graph._mNodes, bGraphTime, eGraphTime)
        self.writeEdge(self._graph._mEdges, bGraphTime, eGraphTime)
        self.writeVicinity(self._graph._vicinity, bGraphTime, eGraphTime)

    def createFolder(self, bGraphTime, eGraphTime):
        filePath = "../graphsData/Graph"+str(bGraphTime)+"_"+str(eGraphTime)
        # if the directory does not exist, create new one
        if not os.path.exists(filePath):
            print("Creating a new graph folder:"+filePath)
            result = False
            try:
                os.makedirs(filePath)
                result = True
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            if result:
                print("Graph Folder created!")
    def writeNode(self, mNodes, bGraphTime, eGraphTime):
        filePath = "../graphsData/Graph"+str(bGraphTime)+"_"+str(eGraphTime)+"/Nodes.txt"
        with open(filePath, "wb") as output:
            pickle.dump(mNodes, output, pickle.HIGHEST_PROTOCOL)

    def writeEdge(self, mEdges, bGraphTime, eGraphTime):
        filePath = "../graphsData/Graph"+str(bGraphTime)+"_"+str(eGraphTime)+"/Edges.txt"
        with open(filePath, "wb") as output:
            pickle.dump(mEdges, output, pickle.HIGHEST_PROTOCOL)

    def writeVicinity(self, vicinity, bGraphTime, eGraphTime):
        filePath = "../graphsData/Graph"+str(bGraphTime)+"_"+str(eGraphTime)+"/Vicinity.txt"
        with open(filePath, "wb") as output:
            pickle.dump(vicinity, output, pickle.HIGHEST_PROTOCOL)

    def setGraph(self, eGraph):
        self._graph = eGraph

    def loadEntityGraph(self, nodeFile, edgeFile, vicinityFile):
        self._graph = Graph()
        if not Path(nodeFile).is_file() or not Path(edgeFile) or not Path(vicinityFile):
            return False
        self._graph.loadNodes(nodeFile)
        self._graph.loadEdges(edgeFile)
        self._graph.loadVicinity(vicinityFile)
        if self._graph.getNodeCnt() > 0 and self._graph.getEdgeCnt() > 0:
            print("Loading graph completed!")
            return True
        else:
            return False


    def updateGraph(self,currentTd, deleteTd, insertTd):
        #set for making delted of added nodes of graph
        markForDel = set()
        markForAdd = set()

        # A temporary vicinity that store unchanged nodes
        tempVicinity = self._graph._vicinity
        print("There are total %d nodes that remain in old vicinity at beginning"%len(tempVicinity))

        #searching for nodes that would be affected by deleting
        for d in deleteTd.getTweets():
            entities = d.getEntities()
            for k in range(0, len(entities)-1):
                for j in range(k+1, len(entities)):
                    if entities[k] == entities[j]:
                        continue

                    node1 = entities[k]
                    node2 = entities[j]

                    for key, values in tempVicinity.items():
                        vicinity = values.keys()
                        if node1 in vicinity or node2 in vicinity:
                            markForDel.add(key)

        #delete keywords in old vicinity
        for keyword in markForDel:
            del tempVicinity[keyword]
        print("There are total %d nodes that remain in old vicinity right after deleting")

        # # generate the new graph
        # buff = TweetDatabase()
        # buff = self._td
        # buff.deleteFromHead(deleteTd.size())
        # buff.addAll(insertTd)

        #generate graph heare
        self._graph = Graph()
        self._graph.generateNodes(currentTd.getTweets())
        self._graph.generateEdges(currentTd.getTweets(), False)

        # List nodename for new graph
        listNodeNameOfGraph = set()
        for node in self._graph._mNodes:
            listNodeNameOfGraph.add(node.getName())

        self._graph._vicinity = tempVicinity
        print("There are total %d nodes that remin in old vicinity at beging of insertion"%len(self._graph._vicinity))
        for d in insertTd.getTweets():
            entities = d.getEntities()
            for k in range(0, len(entities) -1):
                for j in range(k+1, len(entities)):
                    if entities[k] == entities[j]:
                        continue

                    node1 = entities[k]
                    node2 = entities[j]

                    for key, values in tempVicinity.items():
                        vicinity = values.keys()
                        if node1 in vicinity or node2 in vicinity:
                            markForAdd.add(key)

        #delete keywords in old vicinity
        for keyword in markForAdd:
            if keyword in self._graph._vicinity:
                del self._graph._vicinity[keyword]

        print("There are total %d nodes that remain in old vicinity after inserting")

        epsilon = self._config["hubseek"]["epsilon"]
        errorBound = self._config["clustream"]["errorBound"]
        pRestart = self._config["clustream"]["pRestart"]

        searcher = Propagator(self._graph)
        #recompute
        cnt = 0
        for nodeName in listNodeNameOfGraph:
            if not (nodeName in self._graph._vicinity):
                neighbors = searcher.search(nodeName, epsilon, pRestart, errorBound)
                self._graph._vicinity[nodeName] = neighbors
                cnt += 1
                if cnt % 100 == 0:
                    print("Finished re-computing vicinity for %d nodes."%cnt)