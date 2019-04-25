import os
############
train_path = "./train_scene/"
test_path = "./test_scene/"
personname_number=0
for fn in os.listdir(train_path):  # fn 表示的是文件名,生成与orl_scene里子文件夹对应的train_scene和
    # test_scene里对应的子文件夹
    personname_number = personname_number + 1
print("训练人总数：" + str(personname_number))
###############
i=0
classes = {i: i for i in range(0, personname_number)}
for fn in os.listdir(train_path):
    classes[i] = fn
    i = i + 1;
print("标签：")
for j in range(0,personname_number):
    print(classes[j])
###############
train_all=0
for (root, dirs, filenames) in os.walk(train_path):
    for filename in filenames:
        train_all=train_all+1
print("train_all总数："+str(train_all))
test_all=0
for (root, dirs, filenames) in os.walk(test_path):
    for filename in filenames:
        test_all=test_all+1
print("test_all总数："+str(test_all))
#################
ID = [0 for i in range(personname_number)]
i = 0
for fn in os.listdir(train_path):  # fn 表示的是文件名,生成与orl_scene里子文件夹对应的train_scene和
    # orl_scene里对应的子文件夹
    ID[i] = fn
    i = i + 1
print("标签：")
for j in range(0,personname_number):
    print(ID[j])
#################