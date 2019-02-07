import sys
import json
import time
from pyspark import SparkContext

input = sys.argv[1]
output = sys.argv[2]
n_partition = int(sys.argv[3])
sc = SparkContext(appName="Task2")
reviewRDD = sc.textFile(input)
curr = time.time()
default = reviewRDD.map(lambda x: json.loads(x))
default.map(lambda x: (x['business_id'], 1))\
    .reduceByKey(lambda x, y: x+y)\
    .sortBy(lambda x: x[1], False)\
    .take(10)
def_exe_time = time.time()-curr
def_n_part = default.getNumPartitions()
def_n_items = default.glom().map(len).collect()


curr = time.time()
customized = reviewRDD.map(lambda x: json.loads(x))\
    .map(lambda x: (x['business_id'], 1))\
    .partitionBy(n_partition)
customized.reduceByKey(lambda x, y: x+y)\
    .sortBy(lambda x: x[1], False)\
    .take(10)
cus_exe_time = time.time()-curr
cus_n_part = customized.getNumPartitions()
cus_n_items = customized.glom().map(len).collect()

res = {}
def_data = {}
def_data['n_partition'] = def_n_part
def_data['n_items'] = def_n_items
def_data['exe_time'] = def_exe_time
res['default'] = def_data
cus_data = {}
cus_data['n_partition'] = cus_n_part
cus_data['n_items'] = cus_n_items
cus_data['exe_time'] = cus_exe_time
res['customized'] = cus_data
res['explanation'] = "The customized method created partition by key, which means items with same key were put into the" \
                     " same partition. This will largely reduce the shuffling time compared with default method."
json_data = json.dumps(res, indent=4)
open(output, "w").write(json_data)
sc.stop()


