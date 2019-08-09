import sys
import time
import queue
from pyspark import SparkContext
from itertools import combinations
from decimal import *

currTime = time.time()
threshold = int(sys.argv[1])
input = sys.argv[2]
betweenness_output = sys.argv[3]
community_output = sys.argv[4]

sc = SparkContext(appName="Girvan-Newman")
RawRDD = sc.textFile(input).map(lambda x: x.split(','))
header = RawRDD.first()
DataRDD = RawRDD.filter(lambda x: x != header)

dic = {}
for x in DataRDD.collect():
    if x[0] not in dic:
        dic[x[0]] = set()
    dic[x[0]].add(x[1])

edgelist = []
dicedge = {}
for x in combinations(dic.keys(), 2):
    if len(dic[x[0]].intersection(dic[x[1]])) >= threshold:
        edgelist.append(sorted(x))
        if x[0] not in dicedge:
            dicedge[x[0]] = set()
        dicedge[x[0]].add(x[1])
        if x[1] not in dicedge:
            dicedge[x[1]] = set()
        dicedge[x[1]].add(x[0])
nodeRDD = sc.parallelize(list(dicedge.keys()))


def GirvanNewman(root, dicedge):
    dicParent = {}
    visited = set()
    q = queue.Queue()
    q.put(root)
    visited.add(root)
    level = 1
    dicLevel = {}
    dicPathCount = {}
    dicPathCount[root] = 1
    while not q.empty():
        size = q.qsize()
        dicLevel[level] = set()
        for i in range(size):
            curr = q.get()
            for child in dicedge[curr]:
                if child in visited:
                    if child in dicLevel[level]:
                        dicParent[child].add(curr)
                        dicPathCount[child] += 1
                else:
                    if child not in dicParent:
                        dicParent[child] = set()
                    if child not in dicPathCount:
                        dicPathCount[child] = 0
                    dicPathCount[child] += 1
                    dicParent[child].add(curr)
                    q.put(child)
                    visited.add(child)
                    dicLevel[level].add(child)
        level += 1
    dicNode = {}
    for i in range(level-1, 0, -1):
        levelNode = dicLevel[i]
        for curr in levelNode:
            if curr not in dicNode:
                dicNode[curr] = 1
            if curr in dicParent:
                parents = dicParent[curr]
                for parent in parents:
                    toAdd = dicNode[curr]*(Decimal(dicPathCount[parent])/sum(dicPathCount[x] for x in parents))
                    if parent not in dicNode:
                        dicNode[parent] = 1
                    dicNode[parent] += toAdd
                    edge = (min(curr, parent), max(curr, parent))
                    yield (edge, float(toAdd))


betweennessRDD = nodeRDD.flatMap(lambda x: GirvanNewman(x, dicedge))\
    .reduceByKey(lambda x, y: x+y)\
    .map(lambda x: (x[0], x[1]/2))\
    .sortBy(lambda x: x[0][1])\
    .sortBy(lambda x: x[0][0])\
    .sortBy(lambda x: x[1], ascending=False)
betweennesslist = betweennessRDD.collect()

f = open(betweenness_output, "w")

for x in betweennesslist:
    f.write(str(x)[1:-1]+'\n')
f.close()


def determineCommunity(edges):
    res = []
    nodeTolabel = {}
    labelTonode = {}
    num = 1
    for node in edges.keys():
        if node not in nodeTolabel:
            dfs(node, edges, num, nodeTolabel, labelTonode)
            num += 1
    for x in labelTonode.keys():
        res.append(list(labelTonode[x]))
    return res


def dfs(node, edges, num, nodeTolabel, labelTonode):
    if node in nodeTolabel:
        return
    nodeTolabel[node] = num
    if num not in labelTonode:
        labelTonode[num] = set()
    labelTonode[num].add(node)
    for next in edges[node]:
        dfs(next, edges, num, nodeTolabel, labelTonode)


dicdegree = {}
for x in dicedge.keys():
    dicdegree[x] = len(dicedge[x])

dicedgecopy = dicedge.copy()
dicdegreecopy = dicdegree.copy()
edgelistcopy = edgelist.copy()
m = len(edgelistcopy)
modularity = 0
iniCommu = determineCommunity(dicedgecopy)
finalComm = iniCommu
prev_community_count = len(iniCommu)
for commu in iniCommu:
    for i in commu:
        for j in commu:
            if j in dicedge[i]:
                A = 1
            else:
                A = 0
            modularity += (A - dicdegreecopy[i] * dicdegreecopy[j] / (2 * m)) / (2 * m)

while len(edgelistcopy) > 0:
    toremove = nodeRDD\
        .flatMap(lambda x: GirvanNewman(x, dicedgecopy))\
        .reduceByKey(lambda x, y: x+y)\
        .map(lambda x: (x[0], x[1]/2))\
        .sortBy(lambda x: x[0][0])\
        .sortBy(lambda x: x[1], ascending=False)\
        .first()
    removeedge = list(toremove[0])
    node1 = removeedge[0]
    node2 = removeedge[1]
    edgelistcopy.remove(removeedge)
    dicedgecopy[node1].remove(node2)
    dicedgecopy[node2].remove(node1)
    tempmodu = 0
    currCommu = determineCommunity(dicedgecopy)
    currCommuCount = len(currCommu)

    for commu in currCommu:
        for i in commu:
            for j in commu:
                if j in dicedge[i]:
                    A = 1
                else:
                    A = 0
                tempmodu += (A - dicdegreecopy[i] * dicdegreecopy[j] / (2 * m)) / (2 * m)

    if tempmodu > modularity:
        modularity = tempmodu
        finalComm = currCommu
    if prev_community_count < currCommuCount and tempmodu < modularity - 0.05:
        break
    prev_community_count = currCommuCount

resRDD = sc.parallelize(finalComm)\
    .map(lambda x: (len(x), [sorted(x)]))\
    .reduceByKey(lambda x, y: x+y)\
    .map(lambda x: (x[0], sorted(x[1])))\
    .sortByKey()
f = open(community_output, "w")
for x in resRDD.collect():
    communities = x[1]
    for community in communities:
        f.write(str(community)[1:-1]+'\n')
f.close()
print("runtime:", time.time()-currTime, "s")
