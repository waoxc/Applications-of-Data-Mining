import sys
import time
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import random
from collections import defaultdict
from itertools import combinations

curr = time.time()
train = sys.argv[1]
test = sys.argv[2]
case = sys.argv[3]
output = sys.argv[4]
sc = SparkContext(appName="Recommendation System")
trainRDD = sc.textFile(train).map(lambda x: x.split(','))
trainheader = trainRDD.first()
trainRDD = trainRDD.filter(lambda x: x != trainheader)
userRDD = trainRDD.map(lambda x: (x[0], 1))\
    .reduceByKey(lambda x, y: x)\
    .sortByKey().map(lambda x: x[0])
userlist = userRDD.collect()
dicUser = {}
count = 0
for x in userlist:
    dicUser[x] = count
    count += 1
businessRDD = trainRDD.map(lambda x: (x[1], 1))\
    .reduceByKey(lambda x, y: x).sortByKey()\
    .map(lambda x: x[0])
businesslist = businessRDD.collect()
dicBusiness = {}
count = 0
for x in businesslist:
    dicBusiness[x] = count
    count += 1


testRDD = sc.textFile(test).map(lambda x: x.split(','))
testheader = testRDD.first()
testRDD = testRDD.filter(lambda x: x != testheader)\
    .filter(lambda x: x[0] in userlist and x[1] in businesslist)


def setBoundary(x):
    if x[2] > 5:
        rating = 5.0
    elif x[2] < 1:
        rating = 1.0
    else:
        rating = x[2]
    return ((x[0], x[1]), rating)


res = []

if case == '1':

    ratings = trainRDD.map(lambda x: Rating(int(dicUser[x[0]]), int(dicBusiness[x[1]]), float(x[2])))
    rank = 2
    numIterations = 5
    model = ALS.train(ratings, rank, numIterations)
    topredict = testRDD.map(lambda x: (int(dicUser[x[0]]), int(dicBusiness[x[1]])))
    groundrate = testRDD.map(lambda x: Rating(int(dicUser[x[0]]), int(dicBusiness[x[1]]), float(x[2])))
    predictions = model.predictAll(topredict).map(setBoundary)
    ratesAndPreds = groundrate.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    RMSE = pow(MSE, 0.5)
    print('RMSE:', RMSE)
    res = predictions.map(lambda x: (userlist[x[0][0]], businesslist[x[0][1]], x[1])).collect()

elif case == '2':

    def average(userD, userid):
        allbus = userD[userid]
        r = [allbus[i] for i in allbus]
        return sum(r)/len(r)

    def averageExcept(userD, userid, busid):
        allbus = userD[userid]
        r = [allbus[i] for i in allbus if i != busid]
        return sum(r)/len(r)

    def pearson(dir1, dir2):
        intersection = [x for x in dir1 if x in dir2]
        if len(intersection) == 0:
            return 0
        ave1 = sum([dir1[x] for x in intersection])/len(intersection)
        ave2 = sum([dir2[x] for x in intersection])/len(intersection)
        dividend = sum([(dir1[i]-ave1)*(dir2[i]-ave2) for i in intersection])
        divisor1 = pow(sum([(dir1[i]-ave1)**2 for i in intersection]), 0.5)
        divisor2 = pow(sum([(dir2[i]-ave2)**2 for i in intersection]), 0.5)
        if dividend == 0:
            return 0
        return dividend/(divisor1*divisor2)

    def findsimilarusers(user, userD, num):
        ran = []
        for x in range(num):
            ran.append(userlist[random.randint(0, len(userlist)-1)])
        similars = [(pearson(userD[user], userD[x]), x) for x in ran if user != x]
        return similars

    def predict(x, userD, num):
        similars = findsimilarusers(x[0], userD, num)
        ave = average(userD, x[0])
        dividend = 0
        divisor = 0
        for i in range(len(similars)):
            curruser = similars[i][1]
            if x[1] in userD[curruser]:
                dividend += (userD[curruser][x[1]]-averageExcept(userD, curruser, x[1]))*similars[i][0]
            else:
                continue
            divisor += abs(similars[i][0])
        toadd = 0
        if dividend != 0:
            toadd = dividend/divisor
        r = ave + toadd
        if r < 1:
            r = 1.0
        if r > 5:
            r = 5.0
        return r

    trainData = trainRDD.map(lambda x: (x[0], x[1], float(x[2])))
    toPredictRDD = testRDD.map(lambda x: (x[0], x[1]))
    userD = defaultdict(dict)
    for (user, bus, rating) in trainData.collect():
        userD[user][bus] = rating
    # dicp = {}
    # for x in userlist:
    #     p = [(pearson(userD[x], userD[y]), y) for y in userD if x != y]
    #     p.sort(reverse=True)
    #     dicp[x] = p[0:5]
    simnum = 50
    predictRDD = toPredictRDD.map(lambda x: ((x[0], x[1]), predict((x[0], x[1]), userD, simnum)))
    groundrate = testRDD.map(lambda x: (x[0], x[1], float(x[2])))
    ratesAndPreds = groundrate.map(lambda r: ((r[0], r[1]), r[2])).join(predictRDD)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    RMSE = pow(MSE, 0.5)
    res = predictRDD.map(lambda x: (x[0][0], x[0][1], x[1])).collect()
    print('RMSE:', RMSE)

elif case == '3':
    def pearson(dir1, dir2):
        intersection = [x for x in dir1 if x in dir2]
        if len(intersection) == 0:
            return 0
        ave1 = sum([dir1[x] for x in intersection])/len(intersection)
        ave2 = sum([dir2[x] for x in intersection])/len(intersection)
        dividend = sum([(dir1[i]-ave1)*(dir2[i]-ave2) for i in intersection])
        divisor1 = pow(sum([(dir1[i]-ave1)**2 for i in intersection]), 0.5)
        divisor2 = pow(sum([(dir2[i]-ave2)**2 for i in intersection]), 0.5)
        if dividend == 0:
            return 0
        return dividend/(divisor1*divisor2)

    def findsimilarbus(bus, user, busD, num):
        # ran = [x for x in busD if user in busD[x]]
        similars = []
        for x in busD:
            if user in busD[x]:
                sim = pearson(busD[bus], busD[x])
                similars.append((sim, x))
                if len(similars) == num:
                    break
        return similars

    def average(busD, busid):
        alluser = busD[busid]
        r = [alluser[i] for i in alluser]
        return sum(r)/len(r)

    def predict(x, busD, num):
        similars = findsimilarbus(x[1], x[0], busD, num)
        dividend = 0
        divisor = 0
        for i in range(len(similars)):
            currbus = similars[i][1]
            if x[0] in busD[currbus]:
                dividend += busD[currbus][x[0]]*similars[i][0]
            else:
                continue
            divisor += abs(similars[i][0])
        r = 0
        if dividend != 0:
            r = dividend/divisor
        if r == 0:
            r = 3.0
        elif r < 1:
            r = 1.0
        elif r > 5:
            r = 5.0
        return r

    def better(x):
        diff1 = abs(x[1][0][0]-x[1][0][1])
        diff2 = abs(x[1][0][0]-x[1][1])
        if diff1 > diff2:
            bet = x[1][1]
        else:
            bet = x[1][0][1]
        return bet

    trainData = trainRDD.map(lambda x: (x[0], x[1], float(x[2])))
    toPredictRDD = testRDD.map(lambda x: (x[0], x[1]))
    busD = defaultdict(dict)
    for (user, bus, rating) in trainData.collect():
        busD[bus][user] = rating
    simnum = 20
    predictRDD1 = toPredictRDD.map(lambda x: ((x[0], x[1]), predict((x[0], x[1]), busD, simnum)))
    predictRDD2 = toPredictRDD.map(lambda x: ((x[0], x[1]), average(busD, x[1])))
    groundrate = testRDD.map(lambda x: (x[0], x[1], float(x[2])))
    ratesAndPreds = groundrate.map(lambda r: ((r[0], r[1]), r[2])).join(predictRDD1)
    ratesAndPreds = ratesAndPreds.join(predictRDD2)
    # print(ratesAndPreds.take(5))
    predictRDD = ratesAndPreds.map(lambda r: (r[0], better(r)))
    ratesAndPreds = groundrate.map(lambda r: ((r[0], r[1]), r[2])).join(predictRDD)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    RMSE = pow(MSE, 0.5)
    res = predictRDD.map(lambda x: (x[0][0], x[0][1], x[1])).collect()
    print('RMSE:', RMSE)

elif case == '4':

    def pearson(dir1, dir2):
        intersection = [x for x in dir1 if x in dir2]
        if len(intersection) == 0:
            return 0
        ave1 = sum([dir1[x] for x in intersection])/len(intersection)
        ave2 = sum([dir2[x] for x in intersection])/len(intersection)
        dividend = sum([(dir1[i]-ave1)*(dir2[i]-ave2) for i in intersection])
        divisor1 = pow(sum([(dir1[i]-ave1)**2 for i in intersection]), 0.5)
        divisor2 = pow(sum([(dir2[i]-ave2)**2 for i in intersection]), 0.5)
        if dividend == 0:
            return 0
        return dividend/(divisor1*divisor2)

    def average(busD, busid):
        alluser = busD[busid]
        r = [alluser[i] for i in alluser]
        return sum(r)/len(r)

    def predict(x, similarD, busD):
        if x[1] not in similarD:
            r = average(busD, x[1])
        else:
            similars = similarD[x[1]]
            dividend = 0
            divisor = 0
            for i in range(len(similars)):
                currbus = similars[i]
                pear = pearson(busD[x[1]], busD[currbus])
                if x[0] in busD[currbus]:
                    dividend += busD[currbus][x[0]]*pear
                else:
                    continue
                divisor += abs(pear)
            r = 0
            if dividend != 0:
                r = dividend/divisor
        if r == 0:
            r = 3.0
        elif r < 1:
            r = 1.0
        elif r > 5:
            r = 5.0
        return r

    def minhash(x, hash):
        # h(x) = (ax+b)%m
        a = hash[0]
        b = hash[1]
        return min([(a * y + b) % m for y in x])

    def band(x):
        res = []
        for i in range(b):
            res.append(((i, tuple(x[1][i * r:i * r + r])), [x[0]]))
        return res


    def jaccard(x):
        set1 = set(matrixDir[x[0]])
        set2 = set(matrixDir[x[1]])
        inter = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = inter / union
        return (x[0], x[1], jaccard)


    trainData = trainRDD.map(lambda x: (x[0], x[1], float(x[2])))
    toPredictRDD = testRDD.map(lambda x: (x[0], x[1]))
    busD = defaultdict(dict)
    for (user, bus, rating) in trainData.collect():
        busD[bus][user] = rating
    matrixRDD = trainRDD.map(lambda x: (x[1], [dicUser[x[0]]])) \
        .reduceByKey(lambda x, y: x + y).sortByKey()
    matrix = matrixRDD.collect()
    matrixDir = {}
    for x in matrix:
        matrixDir[x[0]] = x[1]

    m = len(userlist)

    hashing = [[7, 97], [59, 263], [337, 821], [163, 409], [257, 647], [1151, 1697],
               [941, 1151], [443, 1231], [743, 2293], [1571, 727], [3709, 2273], [5101, 23],
               [6947, 1931], [8039, 2789], [773, 6661], [4871, 6709], [9973, 3221], [9001, 9511]]

    sigRDD = matrixRDD.map(lambda x: (x[0], [minhash(x[1], hash) for hash in hashing]))
    b = 9
    r = int(len(hashing) / b)
    candidateRDD = sigRDD.flatMap(band) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: len(x[1]) > 1) \
        .flatMap(lambda x: list(combinations(x[1], 2))) \
        .distinct()
    resRDD = candidateRDD.map(lambda x: jaccard(x)) \
        .sortBy(lambda x: x[2], ascending=False)
    similarD = {}
    for x in resRDD.collect():
        bus1 = x[0]
        bus2 = x[1]
        if bus1 in similarD:
            similarD[bus1].append(bus2)
        else:
            similarD[bus1] = [bus2]
        if bus2 in similarD:
            similarD[bus2].append(bus1)
        else:
            similarD[bus2] = [bus1]
    simnum = 1000
    predictRDD = toPredictRDD.map(lambda x: ((x[0], x[1]), predict((x[0], x[1]), similarD, busD)))
    groundrate = testRDD.map(lambda x: (x[0], x[1], float(x[2])))
    ratesAndPreds = groundrate.map(lambda r: ((r[0], r[1]), r[2])).join(predictRDD)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    RMSE = pow(MSE, 0.5)
    res = predictRDD.map(lambda x: (x[0][0], x[0][1], x[1])).collect()
    print('RMSE:', RMSE)

f = open(output, "w")
f.write("user_id, business_id, prediction\n")
for x in res:
    f.write(x[0])
    f.write(',')
    f.write(x[1])
    f.write(',')
    f.write(str(x[2]))
    f.write('\n')
f.close()
print('runtime:', time.time()-curr, 's')
