
class ScoreCell(object):
    def __init__(self, id, score):
        self._id = id
        self._score = score

    def getId(self):
        return self._id

    def getScore(self):
        return self._score

    def __str__(self):
        return "[" + id + "," + self._score + "]"

    def __hash__(self):
        return hash(int(id))

    def __lt__(self, other):
        return self._score < other._score

    def __eq__(self, other):
        if not isinstance(other,type(self)):
            return False
        if other.getId() == self._id:
            return True
        else:
            return False
