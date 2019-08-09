import datetime
import binascii
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import sys
import json

port = int(sys.argv[1])
output = sys.argv[2]
total_hash_index = set()
seen = set()
sc = SparkContext(appName='Bloom')
ssc = StreamingContext(sc, 10)
hashing = [[7, 97], [59, 263], [337, 821], [163, 409], [257, 647], [1151, 1697]]
TN = 0
FP = 0
cities = list()
f = open(output, 'a')
f.write('Time,FPR\n')
f.write(str(datetime.datetime.now())[:19]+','+'0.0\n')
f.close()


def hash(cities):
    for city in cities:
        print(city)
        integer = int(binascii.hexlify(city.encode('utf8')), 16)
        curr_hash_index = set()
        global total_hash_index, seen, TN, FP
        for i in range(len(hashing)):
            a = hashing[i][0]
            b = hashing[i][1]
            hash_result = (a*integer+b) % 200
            curr_hash_index.add(hash_result)
        if curr_hash_index & total_hash_index != curr_hash_index:
            total_hash_index |= curr_hash_index
            TN += 1
        else:
            if city not in seen:
                FP += 1
        seen.add(city)
        print(seen)
    return (FP, TN)


def collect(rdd):
    global cities
    cities = rdd.collect()
    for city in cities:
        hash(city)
    f = open(output, 'a')
    f.write(str(datetime.datetime.now())[:19] + ',' + str(FP/(FP+TN)) + '\n')
    f.close()


batch = ssc.socketTextStream("localhost", port)
lines = batch.map(lambda x: json.loads(x))\
    .map(lambda x: (1, [x['city']]))\
    .reduceByKey(lambda x, y: x+y)\
    .map(lambda x: x[1])\
    .foreachRDD(lambda rdd: collect(rdd))

ssc.start()
ssc.awaitTermination()
