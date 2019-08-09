import sys
import time
from pyspark import SparkContext
from collections import defaultdict
import json
import os

curr = time.time()
path = sys.argv[1]
test = sys.argv[2]
output = sys.argv[3]
train_file = os.path.join(path, 'yelp_train.csv')
user_file = os.path.join(path, 'user.json')
bus_file = os.path.join(path, 'business.json')

sc = SparkContext(appName="Competition")
trainRDD = sc.textFile(train_file).map(lambda x: x.split(','))
train_header = trainRDD.first()
trainRDD = trainRDD.filter(lambda x: x != train_header)
testRDD = sc.textFile(test).map(lambda x: x.split(','))
test_header = testRDD.first()
testRDD = testRDD.filter(lambda x: x != test_header)

dic_ave_user = dict()
dic_ave_bus = dict()
user_RDD = sc.textFile(user_file)\
    .map(lambda x: json.loads(x))\
    .map(lambda x: (x['user_id'], float(x['average_stars'])))
for x in user_RDD.collect():
    dic_ave_user[x[0]] = x[1]
bus_RDD = sc.textFile(bus_file)\
    .map(lambda x: json.loads(x))\
    .map(lambda x: (x['business_id'], float(x['stars'])))
for x in bus_RDD.collect():
    dic_ave_bus[x[0]] = x[1]

res = list()
trainData = trainRDD.map(lambda x: (x[0], x[1], float(x[2])))
toPredictRDD = testRDD.map(lambda x: (x[0], x[1]))
userD = defaultdict(dict)
busD = defaultdict(dict)
for (user, bus, rating) in trainData.collect():
    userD[user][bus] = rating
    busD[bus][user] = rating


def ave(dir):
    ratings = [dir[i] for i in dir]
    if len(ratings) == 0:
        return 0
    else:
        return sum(ratings)/len(ratings)


def ave_Except(dir, id):
    ratings = [dir[i] for i in dir if i != id]
    return sum(ratings) / len(ratings)


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


def find_similar_users(user_id, business_id, userD, busD, num):
    if user_id not in userD or business_id not in busD:
        return list()
    user_poll = busD[business_id].keys()
    list_users = list()
    for user in user_poll:
        list_users.append((user, pearson(userD[user], userD[user_id])))
    similars = sorted(list_users, key=lambda x: x[1], reverse=True)
    if num < len(similars):
        top = similars[:num]
    else:
        top = similars
    return top


def find_similar_items(user_id, business_id, userD, busD, num):
    if user_id not in userD or business_id not in busD:
        return list()
    item_poll = userD[user_id].keys()
    list_items = list()
    for item in item_poll:
        list_items.append((item, pearson(busD[item], busD[business_id])))
    similars = sorted(list_items, key=lambda x: x[1], reverse=True)
    if num < len(similars):
        top = similars[:num]
    else:
        top = similars
    return top


def user_based(user_id, business_id, userD, busD, num, dic_ave_bus):
    similars = find_similar_users(user_id, business_id, userD, busD, num)
    if user_id not in userD:
        average = 0
    else:
        average = ave(userD[user_id])
    if average == 0:
        rating = dic_ave_bus[business_id]
    else:
        dividend = 0
        divisor = 0
        for x in similars:
            curr_user = x[0]
            curr_pearson = x[1]
            dividend += (userD[curr_user][business_id]-ave_Except(userD[curr_user], business_id))*curr_pearson
            divisor += abs(curr_pearson)
        toadd = 0
        if dividend != 0:
            toadd = dividend / divisor
        rating = average + toadd
    if rating < 1.0:
        rating = 1.0
    elif rating > 5.0:
        rating = 5.0
    return rating


def item_based(user_id, business_id, userD, busD, num, dic_ave_user):
    similars = find_similar_items(user_id, business_id, userD, busD, num)
    dividend = 0
    divisor = 0
    for x in similars:
        curr_item = x[0]
        curr_pearson = x[1]
        dividend += busD[curr_item][user_id]*curr_pearson
        divisor += abs(curr_pearson)
    if dividend != 0:
        rating = dividend/divisor
    else:
        rating = dic_ave_user[user_id]
    if rating < 1.0:
        rating = 1.0
    elif rating > 5.0:
        rating = 5.0
    return rating


def ave_based_bus(business_id, busD):
    users_rated_bus = busD[business_id]
    bus_ave = ave(users_rated_bus)
    return bus_ave


def ave_based_user(user_id, userD):
    bus_rated_by_user = userD[user_id]
    user_ave = ave(bus_rated_by_user)
    return user_ave


def final_predict(x, predicts):
    r = (predicts[2]+predicts[3])/2
    if abs(predicts[2]-predicts[3]) > 0.5:
        if abs(predicts[0]-predicts[1]) < 1.0:
            r = (predicts[0]+predicts[1])/2
        else:
            diff0 = abs(predicts[0]-x)
            diff1 = abs(predicts[1]-x)
            diff2 = abs(predicts[2]-x)
            diff3 = abs(predicts[3]-x)
            diff_list = [abs(predicts[i]-x) for i in range(4)]
            diff_min = min(diff_list)
            if diff_min == diff0:
                r = predicts[0]
            elif diff_min == diff1:
                r = predicts[1]
            elif diff_min == diff2:
                r = predicts[2]
            elif diff_min == diff3:
                r = predicts[3]
    return r


neigh_count = 10
predictRDD = toPredictRDD.map(lambda x: ((x[0], x[1]),
                                         (user_based(x[0], x[1], userD, busD, neigh_count, dic_ave_bus),
                                          item_based(x[0], x[1], userD, busD, neigh_count, dic_ave_user),
                                          dic_ave_user[x[0]],
                                          dic_ave_bus[x[1]])))
groundRDD = testRDD.map(lambda x: ((x[0], x[1]), float(x[2])))
pre_of_four = predictRDD.collect()
ground_res = groundRDD.collect()
ratesAndPreds = groundRDD.join(predictRDD)\
    .map(lambda x: (x[0], (final_predict(x[1][0], x[1][1]), x[1][0])))
predict_resRDD = ratesAndPreds.map(lambda x: (x[0], x[1][0]))
less_than_1 = ratesAndPreds.map(lambda r: abs(r[1][0] - r[1][1])).filter(lambda x: x < 1).count()
less_than_2 = ratesAndPreds.map(lambda r: abs(r[1][0] - r[1][1])).filter(lambda x: 1 <= x < 2).count()
less_than_3 = ratesAndPreds.map(lambda r: abs(r[1][0] - r[1][1])).filter(lambda x: 2 <= x < 3).count()
less_than_4 = ratesAndPreds.map(lambda r: abs(r[1][0] - r[1][1])).filter(lambda x: 3 <= x < 4).count()
larger_than_4 = ratesAndPreds.map(lambda r: abs(r[1][0] - r[1][1])).filter(lambda x: x > 4).count()
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
RMSE = pow(MSE, 0.5)
res = predict_resRDD.map(lambda x: (x[0][0], x[0][1], x[1])).collect()
print('\nError Distribution:')
print('>=0 and <1:', less_than_1)
print('>=1 and <2:', less_than_2)
print('>=2 and <3:', less_than_3)
print('>=3 and <4:', less_than_4)
print('>=4:', larger_than_4)
print('\nRMSE:')
print(RMSE)
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
print('\nExecution Time:')
print(time.time()-curr)
