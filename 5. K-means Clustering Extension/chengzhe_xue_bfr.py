from sklearn.cluster import KMeans
import numpy as np
import random
import sys
import time

currTime = time.time()
input = sys.argv[1]
clusterNo = int(sys.argv[2])
output = sys.argv[3]

index_to_dimen = {}
dimen_to_index = {}
groundTruth = {}

f = open(input, "r")
for line in f:
    lines = line.split(',')
    index = int(lines[0])
    label = int(lines[1])
    dimension = []
    for i in range(2, len(lines)):
        dimension.append(float(lines[i]))
    index_to_dimen[index] = dimension
    dimen_to_index[tuple(dimension)] = index
    if label not in groundTruth:
        groundTruth[label] = set()
    groundTruth[label].add(index)
f.close()

d = len(index_to_dimen[0])
n_sample = len(index_to_dimen)
indexes = [i for i in range(n_sample)]
random.shuffle(indexes)
percent = 0.2
init_index = indexes[:int(n_sample*percent)]
DS = {}
CS = list()
RS = set()
DS_summary = {}
CS_summary = {}
for i in range(d):
    DS_summary[i] = list()
    DS[i] = set()
f = open(output, 'w')
f.write('The intermediate results:\n')


def generate_cs_summary(cs_list):
    CS_summary.clear()
    i = 0
    for set in cs_list:
        N = len(set)
        sum = np.zeros(d)
        sumsq = np.zeros(d)
        for x in set:
            dimen = index_to_dimen[x]
            for di in range(d):
                sum[di] += dimen[di]
                sumsq[di] += pow(dimen[di], 2)
        CS_summary[i] = [N, sum, sumsq]
        i += 1


def initialize(init_index):
    # print(len(init_index))
    print('Running Round 1...')
    X = np.array([index_to_dimen[x] for x in init_index])
    kmeans = KMeans(n_clusters=clusterNo*10)
    kmeans.fit(X)
    cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
    for i in cluster:
        if len(cluster[i]) < 10:
            for j in cluster[i]:
                RS.add(init_index[j])
    rest_to_kmean = list(set(init_index)-RS)
    # print(len(rest_to_kmean))
    X = np.array([index_to_dimen[x] for x in rest_to_kmean])
    kmeans = KMeans(n_clusters=clusterNo)
    kmeans.fit(X)
    cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
    for i in cluster:
        for j in cluster[i]:
            DS[i].add(rest_to_kmean[j])
    for i in DS:
        N = len(DS[i])
        sum = np.zeros(d)
        sumsq = np.zeros(d)
        for x in DS[i]:
            dimen = index_to_dimen[x]
            for di in range(d):
                sum[di] += dimen[di]
                sumsq[di] += pow(dimen[di], 2)
        DS_summary[i] = [N, sum, sumsq]
    if len(RS) < 2:
        return
    retained_list = list(RS)
    X = np.array([index_to_dimen[x] for x in RS])
    kmeans = KMeans(n_clusters=int(0.6*len(RS)))
    kmeans.fit(X)
    RS.clear()
    cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
    for i in cluster:
        if len(cluster[i]) == 1:
            RS.add(retained_list[cluster[i][0]])
        else:
            CS_set = set()
            for j in cluster[i]:
                CS_set.add(retained_list[j])
            CS.append(CS_set)
    generate_cs_summary(CS)


def inter_write(round, DS, CS, RS):
    totalDP = 0
    for i in DS:
        totalDP += len(DS[i])
    totalCinCS = len(CS)
    totalCP = 0
    for i in CS:
        totalCP += len(i)
    totalRP = len(RS)
    f.write('Round %d: %d,%d,%d,%d\n' % (round, totalDP, totalCinCS, totalCP, totalRP))


def Mahalanobis(summary, dimension):
    sum = 0
    for i in range(d):
        ci = summary[1][i]/summary[0]
        sigmai = pow(summary[2][i]/summary[0]-ci**2, 0.5)
        sum += ((dimension[i]-ci)/sigmai)**2
    return pow(sum, 0.5)


def merge(cs_list):
    setNo = len(cs_list)
    dic_merge = {}
    toRemove = list()
    visited = set()
    for i in range(setNo):
        if i in visited:
            continue
        for j in range(i+1, setNo):
            if j in visited:
                continue
            data1 = CS_summary[i]
            data2 = CS_summary[j]
            point = data1[1]/data1[0]
            dis = Mahalanobis(data2, point)
            if dis < 2*pow(d, 0.5):
                if i not in dic_merge:
                    dic_merge[i] = set()
                    toRemove.append(cs_list[i])
                    visited.add(i)
                dic_merge[i].add(j)
                toRemove.append(cs_list[j])
                visited.add(j)
    for key in dic_merge:
        toAdd = cs_list[key]
        for i in dic_merge[key]:
            toAdd = toAdd | cs_list[i]
        cs_list.append(toAdd)
    for remove in toRemove:
        cs_list.remove(remove)


def bfr(curr_index):
    for index in curr_index:
        dimen = index_to_dimen[index]
        min_dis = sys.maxsize
        min_clus = 0
        for x in DS_summary:
            dis = Mahalanobis(DS_summary[x], dimen)
            if dis < min_dis:
                min_dis = dis
                min_clus = x
        if min_dis < 2*pow(d, 0.5):
            DS[min_clus].add(index)
            DS_summary[min_clus][0] += 1
            for i in range(d):
                DS_summary[min_clus][1][i] += dimen[i]
                DS_summary[min_clus][2][i] += pow(dimen[i], 2)
        else:
            min_dis_CS = sys.maxsize
            min_clus_CS = 0
            for x in CS_summary:
                dis = Mahalanobis(CS_summary[x], dimen)
                if dis < min_dis_CS:
                    min_dis_CS = dis
                    min_clus_CS = x
            if min_dis_CS < 2*pow(d, 0.5):
                CS[min_clus_CS].add(index)
                CS_summary[min_clus_CS][0] += 1
                for i in range(d):
                    CS_summary[min_clus_CS][1][i] += dimen[i]
                    CS_summary[min_clus_CS][2][i] += pow(dimen[i], 2)
            else:
                RS.add(index)
    if len(RS) < 2:
        return
    retained_list = list(RS)
    X = np.array([index_to_dimen[x] for x in RS])
    kmeans = KMeans(n_clusters=int(0.6 * len(RS)))
    kmeans.fit(X)
    RS.clear()
    cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
    for i in cluster:
        if len(cluster[i]) == 1:
            RS.add(retained_list[cluster[i][0]])
        else:
            CS_set = set()
            for j in cluster[i]:
                CS_set.add(retained_list[j])
            CS.append(CS_set)
    generate_cs_summary(CS)
    merge(CS)
    generate_cs_summary(CS)


def finish():
    removeIndex = list()
    for i in CS_summary:
        summary = CS_summary[i]
        dimen = summary[1]/summary[0]
        min_dis = sys.maxsize
        min_clus = 0
        for x in DS_summary:
            dis = Mahalanobis(DS_summary[x], dimen)
            if dis < min_dis:
                min_dis = dis
                min_clus = x
        if min_dis < 2 * pow(d, 0.5):
            for index in CS[i]:
                DS[min_clus].add(index)
                DS_summary[min_clus][0] += 1
                for j in range(d):
                    DS_summary[min_clus][1][j] += dimen[j]
                    DS_summary[min_clus][2][j] += pow(dimen[j], 2)
            removeIndex.append(i)
    for i in removeIndex:
        CS.pop(i)


def getLabel(index):
    res = -1
    for i in DS:
        if index in DS[i]:
            return i
    return res


initialize(init_index)
inter_write(1, DS, CS, RS)
start = 0
end = int(n_sample*percent)
round = 2

while round <= 5:
    start += int(n_sample * percent)
    end += int(n_sample * percent)
    if round == 5:
        curr_index = indexes[start:]
    else:
        curr_index = indexes[start: end]
    print('Running Round %d...' % round)
    bfr(curr_index)
    if round == 5:
        finish()
    inter_write(round, DS, CS, RS)
    round += 1


cluster_res = {}
ground_res = {}
f.write('\nThe clustering results:\n')
for i in range(n_sample):
    if getLabel(i) not in cluster_res:
        cluster_res[getLabel(i)] = 0
    cluster_res[getLabel(i)] += 1
    f.write('%d,%d\n' % (i, getLabel(i)))
f.close()

for i in groundTruth:
    ground_res[i] = len(groundTruth[i])
print('ground truth:', ground_res)
print('clstering result:', cluster_res)

# label_true = [ground_res[i] for i in ground_res]
# label_pred = [cluster_res[i] for i in cluster_res]
# print(normalized_mutual_info_score(label_true, label_pred))

correct = 0
total = 0
for i in range(clusterNo):
    max_intersection = 0
    for j in range(clusterNo):
        max_intersection = max(max_intersection, len(DS[i] & groundTruth[j]))
    correct += max_intersection
    total += len(groundTruth[i])

print('accuracy:', correct/total)
print('runtime:', time.time()-currTime, 's')
