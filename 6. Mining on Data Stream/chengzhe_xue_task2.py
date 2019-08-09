import datetime
import binascii
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import sys
import json

port = int(sys.argv[1])
output = sys.argv[2]
sc = SparkContext(appName='Flajolet-Martin')
ssc = StreamingContext(sc, 5)
cities = list()
hashing = [[7, 97], [59, 263], [337, 821], [163, 409], [257, 647], [1151, 1697]]
currWindow = 0
res = 0
f = open(output, 'a')
f.write('Time,Ground Truth,Estimation\n')
f.close()


def countTrailingZero(x):
    if x == 0:
        return 0
    count = 0
    while x & 1 == 0:
        x = x >> 1
        count += 1
    return count


def FM(window):
    window = set(window)
    global currWindow, res
    currWindow = len(window)
    print(currWindow)
    estSum = 0
    for i in range(len(hashing)):
        a = hashing[i][0]
        b = hashing[i][1]
        longest = 0
        for city in window:
            integer = int(hash(city))
            hash_result = (a * integer + b) % 512
            zeros = countTrailingZero(hash_result)
            longest = max(longest, zeros)
        currEst = pow(2, longest)
        estSum += currEst
    res = int(estSum/len(hashing))
    print(res)


def collect(rdd):
    global cities
    cities = rdd.collect()
    for city in cities:
        FM(city)
    f = open(output, 'a')
    f.write(str(datetime.datetime.now())[:19] + ',' + str(currWindow) + ',' + str(res) + '\n')
    f.close()


batch = ssc.socketTextStream("localhost", port)
lines = batch.map(lambda x: json.loads(x))\
    .map(lambda x: (1, [x['city']]))\
    .reduceByKey(lambda x, y: x+y)\
    .map(lambda x: x[1])\
    .reduceByWindow(lambda x, y: x+y, None, 30, 10)\
    .foreachRDD(lambda rdd: collect(rdd))

ssc.start()
ssc.awaitTermination()
