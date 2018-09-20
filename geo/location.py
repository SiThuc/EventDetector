import math
from geo.ArrayRealVector import ArrayRealVector
class Location(object):
    EARTH_RADIUS = 6378.137

    def __init__(self, lng, lat):
        self._lng = lng
        self._lat = lat

    def locationFromCoor(self, coordinates):
        if len(coordinates) != 2:
            print("Warning! The size of the coordiantes is not 2!")
        self._lng = coordinates[0]
        self._lat = coordinates[1]

    def getLng(self):
        return self._lng

    def getLat(self):
        return self._lat

    def toRealVector(self):
        return ArrayRealVector(self._lng, self._lat)


    def calcEuclideanDist(self, l):
        latDiff = self._lat - l.getLat()
        lngDiff = self._lng - l.getLng()
        return math.sqrt(math.pow(latDiff, 2)+math.pow(lngDiff, 2))

    # get the Geographical distance to another locations, in kilometer
    def calcGeographicDist(self, l):
        lng1 = self._lng
        lat1 = self._lat
        lng2 = l._lng
        lat2 = l._lat
        radLat1 = self.rad(lat1)
        radLat2 = self.rad(lat2)

        a = radLat1 -radLat2
        b = self.rad(lng1) - self.rad(lng2)
        s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2),2) + math.cos(radLat1)*math.cos(radLat2)*math.pow(math.sin(b/2),2)))
        return s * self.EARTH_RADIUS

    def rad(self, d):
        return d * math.pi / 180.0

    def __str__(self):
        return '[' + str(self._lng) +',' + str(self._lat) + ']'

    def __hash__(self):
        return hash(self._lat) * hash(self._lng)

    def __eq__(self, other):
        if  not isinstance(other, Location):
            return False
        loc =  Location(other)
        if loc.getLat() == self.getLat() and loc.getLng()== self.getLng():
            return True
        else:
            return False

    # def toString(self):
    #     return '[' + self.lng + ', ' + self.lat + ']'

    # def java_hashCode(s):
    #     h = 0
    #     for c in s:
    #         h = (31 * h + ord(c)) & 0xFFFFFFFF
    #     return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000






















