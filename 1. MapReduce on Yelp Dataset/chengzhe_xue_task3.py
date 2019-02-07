import sys
import json
import time
from pyspark import SparkContext

input_rev = sys.argv[1]
input_bus = sys.argv[2]
output_file = sys.argv[3]
output_json = sys.argv[4]
sc = SparkContext(appName="Task3")
id_stars = sc.textFile(input_rev).map(lambda x: json.loads(x))\
    .map(lambda x: (x['business_id'], x['stars']))
id_city = sc.textFile(input_bus).map(lambda x: json.loads(x))\
    .map(lambda x: (x['business_id'], x['city']))
curr = time.time()
city_star1 = id_city.join(id_stars)\
    .map(lambda x: (x[1][0], int(x[1][1])))\
    .aggregateByKey((0, 0), lambda x, y: (x[0]+y, x[1]+1), lambda x, y: (x[0]+y[0], x[1]+y[1]))\
    .map(lambda x: (x[0], round(x[1][0]/x[1][1], 1)))\
    .sortBy(lambda x: (-x[1], x[0])).collect()
for x in range(10):
    print(city_star1[x])
time_m1 = time.time()-curr
print(time_m1)

curr = time.time()
city_star2 = id_city.join(id_stars)\
    .map(lambda x: (x[1][0], int(x[1][1])))\
    .aggregateByKey((0, 0), lambda x, y: (x[0]+y, x[1]+1), lambda x, y: (x[0]+y[0], x[1]+y[1]))\
    .map(lambda x: (x[0], round(x[1][0]/x[1][1], 1)))\
    .sortBy(lambda x: (-x[1], x[0])).take(10)
for x in range(10):
    print(city_star2[x])
time_m2 = time.time()-curr
print(time_m2)

f = open(output_file, "a")
f.write('city,stars\n')
for records in city_star1:
    f.write(records[0]+','+str(records[1])+'\n')
f.close()
res = {}
res['m1'] = time_m1
res['m2'] = time_m2
res['explanation'] = "The method 1 collect all data then print top 10, but the method 2 only take top 10 and stop," \
                     "then print all top 10. Method 1 takes longer because collecting all other results takes a very " \
                     "long time."
json_data = json.dumps(res, indent=4)
open(output_json, "w").write(json_data)
sc.stop()
