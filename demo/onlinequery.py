from demo.query import Query
from geo.tweetdatabase import TweetDatabase
class OnlineQuery(Query):
    def __init__(self, start, end, refWindowSize, minSup, updateWindow):
        super().__init__(start, end, refWindowSize, minSup)
        self._queryTD = TweetDatabase()
        self._deleteTD = TweetDatabase()
        self._insertTD  = TweetDatabase()
        self._startDeleteTS = self._startTS
        self._endDeleteTS = self._startTS + updateWindow
        self._startInsertTS = self._endTS
        self._endInsertTS = self._endTS + updateWindow

    def updateQuery(self, updateWindow):
        self._startTS += updateWindow
        self._endTS += updateWindow
        self._startRefTS += updateWindow
        self._endRefTS += updateWindow
        self._startDeleteTS += updateWindow
        self._endInsertTS += updateWindow
        self._startInsertTS += updateWindow
        self._endInsertTS += updateWindow

    def getStartDeleteTS(self):
        return self._startDeleteTS

    def getEndDeleteTS(self):
        return self._endDeleteTS

    def getStartInsertTS(self):
        return self._startInsertTS

    def getEndInsertTS(self):
        return self._endInsertTS

    def getDeleteTD(self):
        return self._deleteTD

    def getInsertTD(self):
        return self._insertTD

