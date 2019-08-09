import sys
import time
from typing import List, Any

from pyspark import SparkContext
from collections import Counter


def candidate_double_check(basket, candidates):
    dic = {}
    basket = list(basket)
    for candidate in candidates:
        for bas in basket:
            if set(candidate).issubset(bas):
                if candidate in dic:
                    dic[candidate] += 1
                else:
                    dic[candidate] = 1
    return dic.items()


def APriori(basketRDD, support, totalBasket):
    basket = list(basketRDD)
    threshold = support*((len(basket))/totalBasket)
    res = []
    flatten = [item for sublist in basket for item in sublist]
    countOfEach = Counter(flatten)
    countOfEach = {k: v for k, v in countOfEach.items() if v >= threshold}
    candidate1 = sorted(countOfEach.keys())
    candidate = []
    for x in candidate1:
        x = [x]
        candidate.append(tuple(x))
    res.extend(candidate)

    new_frequent = candidate
    k = 2
    while len(new_frequent) > 0:
        combination = set()
        for i in range(len(new_frequent) - 1):
            for j in range(i + 1, len(new_frequent)):
                itemset1 = new_frequent[i]
                itemset2 = new_frequent[j]
                combined = sorted(list(set(itemset1) | set(itemset2)))
                if len(combined) == k:
                    combination.add(tuple(combined))
                else:
                    break

        new_frequent = []
        dic = {}
        for com in combination:
            com = set(com)
            key = tuple(sorted(com))
            for bas in basket:
                if com.issubset(bas):
                    if key in dic:
                        dic[key] += 1
                    else:
                        dic[key] = 1
        for key in dic:
            if dic[key] >= threshold:
                new_frequent.append(key)
        new_frequent = sorted(new_frequent)
        res.extend(new_frequent)
        k += 1
    return res


curr = time.time()
filter_threshold = int(sys.argv[1])
support = int(sys.argv[2])
input = sys.argv[3]
output = sys.argv[4]
f = open(output, "w")

sc = SparkContext(appName="Task2")
RawRDD = sc.textFile(input).map(lambda x: x.split(','))
header = RawRDD.first()
DataRDD = RawRDD.filter(lambda x: x != header)

basketRDD = DataRDD.map(lambda x: (x[0], x[1]))\
    .groupByKey()\
    .mapValues(set)\
    .filter(lambda x: len(x[1]) > filter_threshold)\
    .sortByKey()\
    .map(lambda x: x[1])

totalBasket = basketRDD.count()
map1 = basketRDD.mapPartitions(lambda basket: APriori(basket, support, totalBasket)).map(lambda x: (x, 1))
reduce1 = map1.reduceByKey(lambda x, y: 1).keys().collect()
intermediate = []
intermediate = sorted(reduce1, key=lambda x: (len(x), x))

f.write('Candidates:\n')
if len(intermediate) > 0:
    itemNum = len(intermediate[0])
    f.write(str(intermediate[0]).replace(',', ''))
    for i in range(1, len(intermediate)):
        currNum = len(intermediate[i])
        if currNum == itemNum:
            f.write(',')
        else:
            f.write('\n\n')
        if currNum == 1:
            f.write(str(intermediate[i]).replace(',', ''))
        else:
            f.write(str(intermediate[i]))
        itemNum = currNum

map2 = basketRDD.mapPartitions(lambda basket: candidate_double_check(basket, reduce1))
reduce2 = map2.reduceByKey(lambda x, y: (x+y))
res = reduce2.filter(lambda x: x[1] >= support)
frequent = res.keys().collect()
frequent = sorted(frequent, key=lambda x: (len(x), x))

f.write('\n\nFrequent Itemsets:\n')
if len(frequent) > 0:
    itemNum = len(frequent[0])
    f.write(str(frequent[0]).replace(',', ''))
    for i in range(1, len(frequent)):
        currNum = len(frequent[i])
        if currNum == itemNum:
            f.write(',')
        else:
            f.write('\n\n')
        if currNum == 1:
            f.write(str(frequent[i]).replace(',', ''))
        else:
            f.write(str(frequent[i]))
        itemNum = currNum

f.close()
print("Duration: ", time.time()-curr, "seconds")
