import math

class Utils(object):

    #*******************  Basic aggregate functions *********************/
    # Find the max value of an array.
    @staticmethod
    def max(data):
        if len(data) == 0:
            print("Error when finding the max value. Array length is 0!")
            exit(1)
        maxValue = data[0]
        for i in range(0, len(data)):
            if data[i] > maxValue:
                maxValue = data[i]
        return maxValue

    # Find the sum of the array
    @staticmethod
    def sum(data):
        if len(data) == 0:
            print("Error when finding the sum. Array length is 0!")
            exit(1)
        sumValue = 0
        for i in range(0, len(data)):
            sumValue += data[i]
        return sumValue

    # Find the sum of the list
    @staticmethod
    def sum(data):
        if len(data) == 0:
            print("Error when finding the sum. List size is 0!")
            exit(1)
        sumValue = 0
        for i in range(0, len(data)):
            sumValue += data[i]
        return sumValue

    #/*******************  For log operations *********************/
    # Find the sum of exp of the array
    @staticmethod
    def expSum(data):
        sumValue = 0
        for i in range(0, len(data)):
            sumValue += math.exp(data[i])
        return sumValue

    # Find the sum of exp of the array, and then take the log
    @staticmethod
    def sumExpLog(data):
        maxValue = Utils.max(data)
        sumValue = 0
        for i in range(0, len(data)):
            sumValue += math.exp(data[i] - maxValue)
        return math.log(sumValue) + maxValue

    # Normalize the array
    @staticmethod
    def normalize(data):
        if len(data) == 0:
            print("Error when normalizing. Array length is 0!")
            exit(1)
        sumValue = Utils.sum(data)
        if sumValue == 0:
            print("Warning: sum of the elements is 0 when normalizing!")
            return
        for i in range(0, len(data)):
            data[i] /= sumValue


    # Input: an array in the log domain; Output: the ratio in the exp domain
    @staticmethod
    def logNormalize(data):
        if len(data) == 0:
            print("Error when doing log-sum-exp. Array length is 0!")
            exit(1)
        maxValue = Utils.max(data)
        for i in range(0, len(data)):
            data[i] = math.exp(data[i] - maxValue)
        Utils.normalize(data)

    #/*******************  For normal distributions *********************/

    # @staticmethod
    # def mean(data):
    #     print("Sum:",Utils.sum(data))
    #     print("Length of data:",len(data))
    #     return Utils.sum(data)/len(data)

    @staticmethod
    def std(data):
        squareSum = 0
        for v in data:
            squareSum += v*v
        m = mean(data)
        return math.sqrt(squareSum/len(data) - m * m)

    @staticmethod
    def getQuantile(z):
        assert (z >= 0 and z <= 1)
        return math.sqrt(2) * Utils.inverseError(2*z - 1)

    @staticmethod
    def inverseError(x):
        z = math.sqrt(math.pi) * x
        res = z / 2

        z2 = z * z
        zProd = z*z2    #z3

        res += (1.0 / 24) * zProd

        zProd *= z2 # z ^ 5
        res += (7.0 / 960) * zProd

        zProd *= z2 # z ^ 7
        res += (127 * zProd) / 80640

        zProd *= z2 # z ^ 9
        res += (4369 * zProd) / 11612160

        zProd *= z2 # z ^ 11
        res += (34807 * zProd) / 364953600

        zProd *= z2 # z ^ 13
        res += (20036983 * zProd) / 797058662400

        return res

    @staticmethod
    def mean(data):
        return


def mean(data):
    return Utils.sum(data)/len(data)


if __name__ == "__main__":
    data = [1,3,4,6,7,3,9]
    k = Utils.std(data)
    print(k)

