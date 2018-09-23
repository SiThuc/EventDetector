import os
import csv
import json
import time
import pickle
import dash
from geo.geotweet import GeoTweet
import nltk
from graph.graph import Graph
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import pandas as pd
from config import Config
from clustream.clustream import Clustream
from clustream.microcluster import MicroCluster
from geo.tweetdatabase import TweetDatabase
from hubseek.batchdetector import BatchDetector
from rank.idfweighter import IDFWeighter
from rank.ranker import Ranker
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

        numInitClusTweets = int(self._config['clustream']['numInitTweets'])
        queryFrameDuration = int(self._config['timespan']['init'])
        refWindowSize = int(self._config['query']['refWindowSize'])
        updateWindow = int(self._config['timespan']['update'])
        minSup = int(self._config['hubseek']['minSup'])

        # check if model is already existed
        self._model = load_model("../classifier/lstm_T6_best_weights.02-0.9507.hdf5")
        self._model.summary()

        # load tokenizer file
        file = open("../classifier/tokenizer.pickle", 'rb')
        self._tokenizer = pickle.load(file)
        file.close()

        # # training phase
        # cnnTraining = CNNTraining()
        # cnnTraining.training(self._config)
        # self._tokenizer = cnnTraining._tokenizer
        # self._model = cnnTraining._model

        #Processing raw tweets if the cleaned file is not created yet
        cleaned_tweets_file = self._config['file']['input']['cleaned_tweets']
        if not os.path.isfile(cleaned_tweets_file):
            CleanRawTweets.procesingRawTweet(self._config)
        else:
            print("The cleaned file was created before!")

        # classification process
        classified_tweets = self._config['file']['input']['classified_tweets']  # storing classified tweets
        writer = open(classified_tweets, 'w')

        initClusTweets = list()  # List of tweets for initClustream

        processedTweets = 0     # number of tweets are processed
        checkRunBatch = False
        startRunBatchFrame = 0
        endRunBatchFrame = 0
        startUpdateFrame = 0
        endUpdateFrame = 0

        with open(cleaned_tweets_file, encoding='utf8') as ctf:                 #read the tweet file line by line
            # read header
            line = ctf.readline()
            line = line.replace('\n', '')
            column_names = line.split(',')

            # collect tweets to init clustream
            for i in range(0, numInitClusTweets):
                line = ctf.readline()
                data = line.split(',')
                data_string = self.processLineData(data)
                geo_tweet = GeoTweet.geoTweetFromAString(data_string)
                initClusTweets.append(geo_tweet)
                startRunBatchFrame = geo_tweet.getTimestamp()
            endRunBatchFrame = startRunBatchFrame + queryFrameDuration      # set startTS and endTS for the Batchrun
            print('Clustream is starting to init....')
            self.initClustream(initClusTweets)
            print('Clustream stating is done!')

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
                    input = dtf.text
                    sequences_test = self._tokenizer.texts_to_sequences(input)
                    x_test_seq = pad_sequences(sequences_test, maxlen=45)
                    y = self._model.predict(x_test_seq)
                    dtf['class'] = y
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
                        self._clustream.update(geo_tweet)               #update clustream
                        if row['class'] > 0.65:
                            if not checkRunBatch:                       # if the tweet is predicted as disaster tweet
                                self._tdb.add(geo_tweet)                # add tweet into tweetdatabase
                                if (self._tdb.getEndTimestamp() >= endRunBatchFrame):
                                    self.trigger2(self._tdb)
                                    checkRunBatch = True
                                    startUpdateFrame = self._tdb.getStartTimestamp() + updateWindow
                                    endUpdateFrame = self._tdb.getEndTimestamp() + updateWindow
                            else:
                                if geo_tweet.getTimestamp() < endUpdateFrame:
                                    self._insert.add(geo_tweet)
                                else:
                                    print("There are %d tweets in insert set"%self._insert.size())
                                    startUpdateFrame = startUpdateFrame + updateWindow
                                    endUpdateFrame = endUpdateFrame + updateWindow
                                    self.update(startUpdateFrame)
                                    #self.update_normal(startUpdateFrame)
                            writer.write(temp +'\n')
                    batch = []
                    dtf = None

                processedTweets += 1
                # if processedTweets == 50000:
                #     writer.close()
                #     break
                if processedTweets % 5000 == 0:
                    ttime = time.time() - start
                    print("- %d Tweets consumed %f seconds."%(processedTweets, ttime))
                    self._clustream.printStats()


        writer.close()

        end = time.time()

        consumed_time = end-start
        print("Time consumed is %f"%consumed_time)

    def processLineData(self, data):
        list_string = []
        list_string.append(data[0]) #id
        list_string.append(data[4]) #timestamp
        list_string.append(data[2]) #longitude
        list_string.append(data[3]) #latitude
        cleaned_text = CleanRawTweets.tweet_cleaner(data[1])    #text
        tokenized = tokenizing(cleaned_text)
        list_string.append(tokenized)
        result = '\t'.join(list_string)
        return result

    def trigger2(self, tdb):
        print("In the trigger function")
        self._db = Database(self._config)
        epsilon = float(self._config['hubseek']['epsilon'])
        errorBound = float(self._config['clustream']['errorBound'])
        pRestart = float(self._config['clustream']['pRestart'])
        self._db.generateEntityGraph(self._tdb, epsilon, errorBound, pRestart)
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
            print("THE INTERFACE IS CREATED, PLEASE RUN THE INTEFACE MODULE!!!")
            time.sleep(60)
            detector.printStats()
        return detector

    def update_normal(self, startNewTimes):
        deletedTweets = self._tdb.deleteFromHeadByTime(startNewTimes)
        print("There are %d tweets after removing" % self._tdb.size())

        for tweet in deletedTweets:
            self._delete.add(tweet)
        print("There are %d tweets are removed"%self._delete.size())

        for tweet in self._insert.getTweets():
            self._tdb.add(tweet)
        print("There are %d tweets after adding" %self._tdb.size())
        print("----------------------------------------")

        epsilon = float(self._config['hubseek']['epsilon'])
        errorBound = float(self._config['clustream']['errorBound'])
        pRestart = float(self._config['clustream']['pRestart'])
        self._db.generateEntityGraph(self._tdb.getTweets(), epsilon, errorBound, pRestart)
        print("Create graph done!")
        hubseek = self.runHubSeek2(self._tdb)

        self._insert = TweetDatabase()
        self._delete = TweetDatabase()

    def update(self, startNewTimes):
        deletedTweets = self._tdb.deleteFromHeadByTime(startNewTimes)
        print("There are %d tweets after removing"%self._tdb.size())
        self._tdb.setNewStartTimestamp(startNewTimes)

        for tweet in deletedTweets:
            self._delete.add(tweet)
        print("There are %d tweets are removed"%self._delete.size())

        for tweet in self._insert.getTweets():
            self._tdb.add(tweet)
        print("There are %d tweets after adding" %self._tdb.size())
        print("----------------------------------------")

        onDetector = self.runOnline(self._tdb, self._delete, self._insert)
        self._insert = TweetDatabase()
        self._delete = TweetDatabase()

    def runOnline(self, currentTd, deleteTd, insertTd):
        self._detector._hubSeek.delete(deleteTd.getTweets())
        self._detector._hubSeek.insert(insertTd.getTweets())

        minSup = int(self._config['hubseek']['minSup'])
        clusters = self._detector._hubSeek.genClusters(minSup)

        # This part for update raking process
        eta = self._config["hubseek"]["eta"][0]
        bandwidth = self._config["hubseek"]["bandwidth"][0]
        refSpanTime = self._config["query"]["refWindowSize"]

        self.updateRanker(self._tdb, eta)
        self._detector.setTD(currentTd)
        events = self._detector.rank(clusters, bandwidth, refSpanTime)
        print("There are %d events in online step"%len(events))
        
        output = '../output/live/output.json'
        #output2 = '../output/output_' + str(currentTd.getStartTimestamp())+'_'+str(currentTd.getEndTimestamp())+'.json'
        data = []
        for clus in events:
            sub = clus.toJson()
            data.append(sub)

        with open(output, 'w') as f:
            json.dump(data, f)
        f.close()

        for clus in events:
            print(clus.__str__())
            print("################################")
        print("Online Hubseek generating candidates done!!!")
        return clusters

    def updateRanker(self, tdb, eta):
        weighter = IDFWeighter(tdb.getTweets())
        self._detector._ranker = Ranker(self._clustream, weighter, eta)


    def initClustream(self, initClusTweets):
        numInitClusters = self._config['clustream']['numInitClusters']
        numMaxClusters = self._config['clustream']['numMaxClusters']
        numTweetPeriod = self._config['clustream']['numTweetPeriod']
        outdatedThreshold = self._config['clustream']['outdatedThreshold']
        self._clustream = Clustream(numMaxClusters, numTweetPeriod, outdatedThreshold)
        self._clustream.init(initClusTweets, numInitClusters)
        print("The Clustream is also initiated................Done")

    def updateTweetDatabase(self, delete, insert):
        self._tdb.deleteFromHead(delete.size())
        self._tdb.addAll(insert)

    def main(self):
        paraFile = "../config/config.yml"
        self.init(paraFile)

if __name__ == "__main__":
    dm = Demo()
    dm.main()
