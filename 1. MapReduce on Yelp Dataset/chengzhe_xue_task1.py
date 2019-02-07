import sys
import json
import time
from pyspark import SparkContext

curr = time.time()
input = sys.argv[1]
output = sys.argv[2]
sc = SparkContext(appName="Task1")
reviewRDD = sc.textFile(input)
data = reviewRDD.map(lambda x: json.loads(x))

review = data.map(lambda x: (x['date'][:4], 1))\
    .reduceByKey(lambda x, y: x+y)
res1 = review.map(lambda x: (1, x[1]))\
    .reduceByKey(lambda x, y: x+y)\
    .collect()
res2 = review.collect()
re_2018 = 0
for x in res2:
    if x[0] == '2018':
        re_2018 = x[1]
        break

user = data.map(lambda x: (x['user_id'], 1))\
    .reduceByKey(lambda x, y: x+y)\
    .sortBy(lambda x: x[1], False)
res3 = user.map(lambda x: (1, 1))\
    .reduceByKey(lambda x, y: x+y)\
    .collect()
res4 = user.take(10)

business = data.map(lambda x: (x['business_id'], 1))\
    .reduceByKey(lambda x, y: x+y)\
    .sortBy(lambda x: x[1], False)
res5 = business.map(lambda x: (1, 1))\
    .reduceByKey(lambda x, y: x+y)\
    .collect()
res6 = business.take(10)

res = {}
res["n_review"] = res1[0][1]
res["n_review_2018"] = re_2018
res["n_user"] = res3[0][1]
res["top10_user"] = res4
res["n_business"] = res5[0][1]
res["top10_business"] = res6
json_data = json.dumps(res, indent=4)
open(output, "w").write(json_data)
sc.stop()
print(time.time()-curr)
