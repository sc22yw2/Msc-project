from glob import glob
import os
import shutil
import random
source_dir = ["../part1/part1","../part2/part2","../part3/part3"]

li = []
for i in source_dir:
    li.extend(glob(i+"/*.jpg"))
    li.extend(glob(i+'/*.png'))

random.shuffle(li)
len_li = len(li)
train_li = li[:int(len(li)*0.8)]
test_li = li[int(len(li)*0.8):]
for i in train_li:
    shutil.move(i, "train")
for i in test_li:
    shutil.move(i, "test")

