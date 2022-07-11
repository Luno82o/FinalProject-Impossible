# 引入 time 模組
import time
import os
import copy
# 現在時間
now = time.ctime()

# 輸出結果
print("現在時間：", now)
#time.sleep(2)
now = time.ctime()
print("現在時間：", now)

#list


dict3={"id":[4,5]}
if 6 not in dict3.get("id"):
    dict3["id"].append(6)
print(dict3)

list1={3:"test"}
list4={1:"dict1",3:"tset2"}
d1=set(list4).difference(set(list1.keys()))
[list4.pop(key) for key in d1] 
print(list4)

os.system('')
a="hello"
print("      |#{}預測結果為： {:>12}|".format((10),"\033[31m"+a+"\033[37m"))

dict5={1:{"cur":"walk","walk":2},5:{"cur":"ss","abc":3}}

#[dict5[key].pop(a) for key in dict5.keys() for a in dict5[key].keys()] 


for key in dict5.keys():
    [dict5[key].pop(a) for a in copy.deepcopy(dict5)[key].keys() if a!="cur"]
    
print(dict5)


