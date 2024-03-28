#encoding=utf-8
from glob import glob
import os
import shutil
import random
from scipy.io import loadmat
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import uuid
random.seed(42)
source_dir = "imdb_source/imdb_crop/"


# 加载MATLAB文件
mat = loadmat('imdb_source/imdb_crop/imdb.mat')

# 查看MAT文件中的所有变量
print(mat.keys())
data = mat["imdb"][0][0]
photo_takens = data["photo_taken"][0]
full_paths = data["full_path"][0]
genders = data["gender"][0]

# shutil.rmtree("imdb")
os.makedirs("imdb",exist_ok=True)
os.makedirs("imdb/train",exist_ok=True)
os.makedirs("imdb/test",exist_ok=True)
train_done_list = glob("imdb/train/*")
test_done_list = glob("imdb/test/*")
train_done_list.extend(test_done_list)
all_done_list = set(train_done_list)
for i,(photo_taken,full_path,gender) in enumerate(tqdm(list(zip(photo_takens,full_paths,genders)))):
    #gender man is 1
    basename = os.path.basename(full_path[0])
    try:
        age_list = re.findall("_(\d{4}).*?_(\d{4})",basename)[0]
    except:
        continue
    age = int(age_list[1]) - int(age_list[0])
    if age <0:
        print("age < 0 ")
        continue
    id = str(i)

    if gender == 0:
        new_basename = f"{age}_{1}_{id}"
    else:
        new_basename = f"{age}_{0}_{id}"
    ran = random.randint(1, 100)
    if ran<80:
        destination_file = 'imdb/train/'+new_basename+".jpg"
    else:
        destination_file = 'imdb/test/'+new_basename+".jpg"
    if destination_file in all_done_list:
        continue
    else:
        shutil.copy(os.path.join(source_dir,full_path[0]), destination_file)



print()