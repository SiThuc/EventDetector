import os
import csv
import time
import pickle
from geo.geotweet import GeoTweet
import nltk
from graph.graph import Graph
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import pandas as pd
from pathlib import Path
from config import Config
from mongo import Mongo
from clustream.clustream import Clustream
from clustream.microcluster import MicroCluster
from geo.tweetdatabase import TweetDatabase
from hubseek.batchdetector import BatchDetector
from database import Database
from keras.models import load_model
from cleanrawtweets import CleanRawTweets
from keras.preprocessing.sequence import pad_sequences
import keras
from cnn_training import CNNTraining

def tokenizing(text):
    tokens = nltk.word_tokenize(text)
    filtered_sentence = [w for w in tokens if not w in stop_words]
    filtered_sentence = []
    for w in tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    tokenized_centence = '\t'.join(filtered_sentence)
    return tokenized_centence

class Demo(object):
    def __init__(self):
        self._db = None
        self._tdb = TweetDatabase()
        self._insert = TweetDatabase()
        self._delete = TweetDatabase()
        self._clustream = None
        self._config = None
        #self._mongo = None

        self._checkInit = False
        self._checkBatchRun = False
        self._tokenizer = None
        self._model = None
        self._detector = None
        print('The program begin')

    def init(self, paraFile):
        start = time.time()
        # Read YAML file
        self._config = Config.load(paraFile)
        print(self._config)

        init_duration = int(self._config['timespan']['init'])
        update_span = int(self._config['timespan']['update'])



        # # training phase
        # cnnTraining = CNNTraining()
        # cnnTraining.training(self._config)
        # self._tokenizer = cnnTraining._tokenizer
        # self._model = cnnTraining._model

        # Processing raw tweets if the cleaned file is not created yet
        cleaned_tweets_file = self._config['file']['input']['cleaned_tweets']
        if not os.path.isfile(cleaned_tweets_file):
            CleanRawTweets.procesingRawTweet(self._config)
        else:
            print("The cleaned file was created before!")

        countInit = 0
        checkInit = False
        startUpdateFrame = 0
        endUpdateFrame = 0

        with open(cleaned_tweets_file, encoding='utf8') as ctf:                 #read the tweet file line by line
            line = ctf.readline()                                               #read header
            line = line.replace('\n', '')
            column_names = line.split(',')
            dtf = None
            batch = []
            while True :
                line = ctf.readline()
                if line == '':
                    break
                line = line.replace('\n', '')
                data = line.split(',')
                batch.append(data)
                if len(batch) == 100:
                    dtf = pd.DataFrame(batch, columns=column_names)
                    for index, row in dtf.iterrows():
                        list_string = []
                        list_string.append(row['id'])
                        list_string.append(row['timestamp'])
                        list_string.append(row['lng'])
                        list_string.append(row['lat'])
                        cleaned_text = CleanRawTweets.tweet_cleaner(row['text'])
                        tokenized = tokenizing(cleaned_text)
                        list_string.append(tokenized)
                        temp = '\t'.join(list_string)
                        geo_tweet = GeoTweet.geoTweetFromAString(temp)

                        if not checkInit:
                            self._tdb.add(geo_tweet)
                            if (self._tdb.getEndTimestamp() - self._tdb.getStartTimestamp() >= init_duration):
                                self.trigger2(self._tdb)
                                checkInit = True
                                startUpdateFrame = self._tdb.getStartTimestamp() + update_span
                                endUpdateFrame = self._tdb.getEndTimestamp() + update_span
                        else:
                            self._clustream.update(geo_tweet)  # update geotweet
                            if geo_tweet.getTimestamp() < endUpdateFrame:
                                self._insert.add(geo_tweet)

                            else:
                                print("There are %d tweets in insert set"%self._insert.size())
                                startUpdateFrame = startUpdateFrame + update_span
                                endUpdateFrame = endUpdateFrame + update_span
                                self.update(startUpdateFrame)
                    batch = []
                    dtf = None

                countInit += 1
                if countInit % 1000 == 0:
                    ttime = time.time() - start
                    print("- %d Tweets consumed %f seconds."%(countInit, ttime))
                if countInit == 20000:
                    break
        end = time.time()

        consumed_time = end-start
        print("Time consumed is %f"%consumed_time)

        # # Load mongo config               This part is under considering
        # self._mongo = Mongo(self._config)

        # initialize database
        #self.initDatabase()
        # self.initClustream()

    def trigger2(self, tdb):
        print('Initializating clustream....')
        self.initClustream()
        print("In the trigger function")
        self._db = Database(self._config)
        epsilon = float(self._config['hubseek']['epsilon'])
        errorBound = float(self._config['clustream']['errorBound'])
        pRestart = float(self._config['clustream']['pRestart'])
        self._db.generateEntityGraph(self._tdb.getTweets(), epsilon, errorBound, pRestart)
        print("Create graph done!")
        self._detector = self.runHubSeek2(self._tdb)

    def runHubSeek2(self, tdb):
        detector = BatchDetector(self._clustream, self._db.getEntityGraph(), self._config)
        if self._config["hubseek"]["run"]:
            bandwidth = self._config["hubseek"]["bandwidth"][0]
            epsilon = self._config["hubseek"]["epsilon"]
            eta = self._config["hubseek"]["eta"][0]
            minSup = int(self._config['hubseek']['minSup'])
            print("Starting Hubseek process.....")
            refSpanTime = self._config["query"]["refWindowSize"]
            detector.detect(tdb, bandwidth, epsilon, minSup, refSpanTime, eta)
            detector.printStats()
        return detector

    def update(self, startNewTimes):
        deletedTweets = self._tdb.deleteFromHeadByTime(startNewTimes)
        print("There are %d tweets after removing"%self._tdb.size())

        for tweet in deletedTweets:
            self._delete.add(tweet)
        print("There are %d tweets are removed"%self._delete.size())

        for tweet in self._insert.getTweets():
            self._tdb.add(tweet)
        print("There are %d tweets after adding" %self._tdb.size())
        print("----------------------------------------")

        #self._db.updateGraph(self._tdb, self._delete, self._insert)
        onDetector = self.runOnline(self._tdb, self._delete, self._insert)

        # epsilon = float(self._config['hubseek']['epsilon'])
        # errorBound = float(self._config['clustream']['errorBound'])
        # pRestart = float(self._config['clustream']['pRestart'])
        # self._db.generateEntityGraph(self._tdb.getTweets(), epsilon, errorBound, pRestart)
        # hubseek = self.runHubSeek2(self._tdb)

        self._insert = TweetDatabase()
        self._delete = TweetDatabase()

    def runOnline(self, currentTd, deleteTd, insertTd):
        self._detector._hubSeek.delete(deleteTd.getTweets())
        self._detector._hubSeek.insert(insertTd.getTweets())
        minSup = int(self._config['hubseek']['minSup'])
        clusters = self._detector._hubSeek.genClusters(minSup)

        for clus in clusters:
            print(clus.__str__())
            print("################################")
        print("Online Hubseek generating candidates done!!!")
        return clusters

    def initClustream(self):
        numInitClusters = self._config['clustream']['numInitClusters']
        numMaxClusters = self._config['clustream']['numMaxClusters']
        numTweetPeriod = self._config['clustream']['numTweetPeriod']
        outdatedThreshold = self._config['clustream']['outdatedThreshold']
        self._clustream = Clustream(numMaxClusters, numTweetPeriod, outdatedThreshold)
        self._clustream.init(self._tdb.getTweets(), numInitClusters)
        print("The Clustream is also initiated................Done")
#-------------------------------------------------------------------------------------------
    def initDatabase(self):
        tweetFile = self._config['file']['input']['tweets']
        numInitTweets = self._config['clustream']['numInitTweets']
        self._db = Database(self._config)
        self._db.loadInitialTweets(tweetFile, numInitTweets)
        print("Load Inital tweets.................Done")



    def runBatch(self):
        epsilon = self._config['hubseek']['epsilon'][0]
        #numInitTweets = self._config['clustream']['numInitTweets']
        pRestart = self._config['clustream']['pRestart']
        errorBound = self._config['clustream']['errorBound']

        queryDB = TweetDatabase()
        refDB = TweetDatabase()

        queries = self._mongo.loadBatchQueries(self._config)
        # # print all queries
        # for qr in queries:
        #     print("StartQueryTS:%d, EndQueryTS:%d"%(qr.getStartTS(), qr.getEndTS()))
        queryIndex = 0
        clustream_path = "../clustream/clustream_data/clustream_" + str(queryIndex) + ".pickle"
        myClustream_file = Path(clustream_path)

        query = queries[queryIndex]
        if myClustream_file.is_file():
            self.load_clustream(clustream_path)
        else:   # update the cluster
            while True:
                tweet = self._db.nextTweet()
                if tweet is None:
                    break
                self.addTweet(query, queryDB, refDB, tweet)
                self._clustream.update(tweet)
                if tweet.getTimestamp() > query.getEndTS():
                    print("There are: %d Tweets in query: %d"%(len(queryDB.getTweets()), queryIndex))
                    print("There are: %d Tweets in refDB: %d"%(len(refDB.getTweets()), queryIndex))

                    #saving current clustream
                    self._clustream.printStats()
                    with open(clustream_path, "wb") as output:
                        pickle.dump(self._clustream, output, pickle.HIGHEST_PROTOCOL)
                    print("The Current _clustream is stored!")

                    graphTime = self._config["query"]["querypoints"]
                    entityFile = "../graphsData/Graph"+str(graphTime[0])+"_"+str(graphTime[1])+"/Nodes.txt"
                    entityEdgeFile = "../graphsData/Graph"+str(graphTime[0])+"_"+str(graphTime[1])+"/Edges.txt"
                    vicinityFile = "../graphsData/Graph"+str(graphTime[0])+"_"+str(graphTime[1])+"/Vicinity.txt"
                    #check whether the entitygraph can be loaded or not
                    check = self._db.loadEntityGraph(entityFile, entityEdgeFile, vicinityFile)
                    if not check:
                        self._db.generateEntityGraph(queryDB.getTweets(), epsilon, errorBound, pRestart)

                    self.trigger(query, queryDB)
                    queryIndex += 1
                    if queryIndex < len(queries):
                        query = queries[queryIndex]
                        queryDB = TweetDatabase()
                        refDB.deleteFromHead(query.getRefStartTS())
                    else:
                        break
            print("Batch mode ..............Done!")

    def load_clustream(self, filePath):
        with open(filePath, "rb") as ip:
            self._clustream = pickle.load(ip)
        print("The clustream is loaded")

    def addTweet(self, query, queryDB, refDB, tweet):
        ts = tweet.getTimestamp()
        if ts > query.getStartTS() and ts <= query.getEndTS():
            queryDB.add(tweet)
        if ts > query.getRefStartTS() and ts <= query.getRefEndTS():
            refDB.add(tweet)

    def trigger(self, query, queryDB):
        if queryDB.size() == 0:
            return
        hubseek = self.runHubSeek(query, queryDB)


    def runHubSeek(self, query, queryDB):
        detector = BatchDetector(self._clustream, self._db.getEntityGraph(), self._config)
        if self._config["hubseek"]["run"]:
            bandwidth = self._config["hubseek"]["bandwidth"][0]
            epsilon = self._config["hubseek"]["epsilon"][0]
            eta = self._config["hubseek"]["eta"][0]
            refTimeSpan = query.getRefEndTS() - query.getRefStartTS()
            minSup = query.getMinSup()
            print("Starting Hubseek process.....")
            detector.detect(queryDB, query.getQueryInterval(), bandwidth, epsilon, minSup, refTimeSpan, eta)
            detector.printStats()
        return detector


    def main(self):
        paraFile = "../config/config.yml"
        self.init(paraFile)
        #self.runBatch()
        # self.runOnlilne()


if __name__ == "__main__":
    dm = Demo()
    dm.main()

