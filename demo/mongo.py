from pymongo import MongoClient
from geo.tweetdatabase import TweetDatabase
from geo.location import Location
from geo.geotweet import GeoTweet
from query import Query

class Mongo(object):
    def __init__(self, config):
        host = config['mongo']['dns']
        port = config['mongo']['port']
        db = config['mongo']['db']
        tweetColName = config['mongo']['tweet_col']

        print("- DB Host:",host)
        print("- DB Port:",port)
        print("- DB name:",db)
        print("- TweetColName:", tweetColName)

        self._client = MongoClient(host=host, port=port)
        self._database = self._client.get_database(db)
        self._tweetCol = self._database.get_collection(tweetColName)

    def rangeQueryTweetDB(self, startTS, endTS):
        cursor = self._tweetCol.find({"timestamp":{"$gte":startTS, "$lt":endTS}})
        td = TweetDatabase()
        for data in cursor:
            tweetId = data["id"]
            timestamp = data["timestamp"]
            lng = data["lng"]
            lat = data["lat"]
            loc = Location(lng, lat)
            entities = data["entities"]
            tweet = GeoTweet(tweetId, timestamp, loc, entities)
            td.append(tweet)
        return td

    def loadBatchQueries(self, config):
        refWindowSize = config["query"]["refWindowSize"]
        minSup = config["query"]["minSup"]
        queries = []
        queryList = config["query"]["querypoints"]
        for i in range(0, len(queryList), 2):
            startTS = queryList[i]
            endTS = queryList[i+1]
            queries.append(Query(startTS, endTS, refWindowSize, minSup))
        return queries


    def loadTweetsIntoQueryDB(self, query, queryDB, refDB):
        # Load for queryDB
        cursor = self._tweetCol.find({"timestamp":{"$gte":query.getStartTS(), "$lt":query.getEndTS()}})
        for data in cursor:
            tweetId = data["id"]
            timestamp = data["timestamp"]
            lng = data["lng"]
            lat = data["lat"]
            loc = Location(lng, lat)
            entities = data["entities"]
            tweet = GeoTweet(tweetId, timestamp, loc, entities)
            queryDB.append(tweet)

        # Load for refDB
        cursor = self._tweetCol.find({"timestamp":{"$gte":query.getRefStartTS(), "$lt":query.getRefEndTS()}})
        for data in cursor:
            tweetId = data["id"]
            timestamp = data["timestamp"]
            lng = data["lng"]
            lat = data["lat"]
            loc = Location(lng, lat)
            entities = data["entities"]
            tweet = GeoTweet(tweetId, timestamp, loc, entities)
            refDB.append(tweet)

# if __name__ == "__main__":
#     client = MongoClient("127.0.0.1", 27017)
#     database = client.get_database("EmployeeData")
#     col = database.get_collection("Employees")
#     doc = col.find({"id":{"$gt":2}, "age": {"$lt":29,"$gt":2}})
#     for x in doc:
#         print(x['name'])