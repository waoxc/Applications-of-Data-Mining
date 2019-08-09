import sys
import json
from pyspark import SparkContext

in_review = sys.argv[1]
in_business = sys.argv[2]
out_csv = sys.argv[3]

sc = SparkContext(appName='generateCSV')
bus_state = sc.textFile(in_business).map(lambda x: json.loads(x))\
    .map(lambda x: (x['business_id'], x['state']))\
    .filter(lambda x: x[1] == 'NV')\
    .map(lambda x: x[0])\
    .collect()
business_set = set(bus_state)
print(business_set)
user_bus = sc.textFile(in_review).map(lambda x: json.loads(x))\
    .map(lambda x: (x['user_id'], x['business_id']))\
    .filter(lambda x: x[1] in business_set)\
    .sortByKey()\
    .collect()
print(len(user_bus))
f = open(out_csv, "w")
f.write('user_id,business_id\n')
for entry in user_bus:
    f.write(entry[0])
    f.write(',')
    f.write(entry[1])
    f.write('\n')
f.close()
