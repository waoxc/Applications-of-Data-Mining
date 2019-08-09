import sys
import time
from pyspark import SparkContext
from itertools import combinations

curr = time.time()
input = sys.argv[1]
output = sys.argv[2]

sc = SparkContext(appName="Jaccrad based LSH")
RawRDD = sc.textFile(input).map(lambda x: x.split(','))
header = RawRDD.first()
DataRDD = RawRDD.filter(lambda x: x != header)

userRDD = DataRDD.map(lambda x: (x[0], 1))\
    .reduceByKey(lambda x, y: x)\
    .sortByKey().map(lambda x: x[0])
userlist = userRDD.collect()
dicUser = {}
count = 0
for x in userlist:
    dicUser[x] = count
    count += 1

business = DataRDD.map(lambda x: (x[1], 1))\
    .reduceByKey(lambda x, y: x).sortByKey()\
    .map(lambda x: x[0])
matrixRDD = DataRDD.map(lambda x: (x[1], [dicUser[x[0]]]))\
    .reduceByKey(lambda x, y: x+y).sortByKey()
matrix = matrixRDD.collect()
matrixDir = {}
for x in matrix:
    matrixDir[x[0]] = x[1]

m = len(userlist)

hashing = [[7, 97], [59, 263], [337, 821], [163, 409], [257, 647], [1151, 1697],
           [941, 1151], [443, 1231], [743, 2293], [1571, 727], [3709, 2273], [5101, 23],
           [6947, 1931], [8039, 2789], [773, 6661], [4871, 6709], [9973, 3221], [9001, 9511]]


def minhash(x, hash):
    # h(x) = (ax+b)%m
    a = hash[0]
    b = hash[1]
    return min([(a*y+b) % m for y in x])


sigRDD = matrixRDD.map(lambda x: (x[0], [minhash(x[1], hash) for hash in hashing]))
b = 9
r = int(len(hashing)/b)


def band(x):
    res = []
    for i in range(b):
        res.append(((i, tuple(x[1][i*r:i*r+r])), [x[0]]))
    return res


candidateRDD = sigRDD.flatMap(band)\
    .reduceByKey(lambda x, y: x+y)\
    .filter(lambda x: len(x[1]) > 1)\
    .flatMap(lambda x: list(combinations(x[1], 2)))\
    .distinct()


def jaccard(x):
    set1 = set(matrixDir[x[0]])
    set2 = set(matrixDir[x[1]])
    inter = len(set1 & set2)
    union = len(set1 | set2)
    jaccard = inter/union
    return (x[0], x[1], jaccard)


resRDD = candidateRDD.map(lambda x: jaccard(x))\
    .filter(lambda x: x[2] >= 0.5)\
    .sortBy(lambda x: x[1])\
    .sortBy(lambda x: x[0])


# truth = sc.textFile("pure_jaccard_similarity.csv").map(lambda x: x.split(','))
# head = truth.first()
# truthdata = truth.filter(lambda x: x != head).map(lambda x: (x[0], x[1]))
# resdata = resRDD.map(lambda x: (x[0], x[1]))
# tp = resdata.intersection(truthdata).collect()
# pre = len(tp)/len(resdata.collect())
# re = len(tp)/len(truthdata.collect())
# print(len(tp), len(truthdata.collect()), pre, re)


f = open(output, "w")
f.write("business_id_1, business_id_2, similarity\n")
for x in resRDD.collect():
    f.write(x[0])
    f.write(',')
    f.write(x[1])
    f.write(',')
    f.write(str(x[2]))
    f.write('\n')
f.close()
print('runtime:', time.time()-curr, 's')
