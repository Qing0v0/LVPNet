# 分配数据集
import numpy as np
import random
import json


SEED = 1234
TRAIN_VAL_PROPORTION = 1 # 除去测试集占多少比例
VAL_PROPORTION = 0.0785       # 验证集占 验证-训练集的多少
# LABEL_PATH = ["depth_train.json", "depth_val.json", "depth_test.json"]
LABEL_PATH = ["depth_train.json", "depth_val.json"]
np.random.seed(SEED)
random.seed(SEED)

def get_divided_list(data_list:list):
    data_num = len(data_list)
    train_val_num = int(data_num * TRAIN_VAL_PROPORTION)
    val_num = int(data_num * TRAIN_VAL_PROPORTION * VAL_PROPORTION)

    train_val_list = random.sample(data_list, train_val_num)
    val_list = random.sample(train_val_list, val_num)
    train_list = list(set(train_val_list) - set(val_list))
    test_list = list(set(data_list) - set(train_val_list))
    
    # return train_list, val_list, test_list
    return train_list, val_list

def write_list(json_list:list, file_name:str) -> None:
    with open(file_name, 'a', encoding='utf-8') as f:
        for json_str in json_list:
            f.writelines(json_str)

if __name__ == "__main__":
    with open(r"depth_label/depth_label.json", "r", encoding="utf-8") as f:
        json_list = f.readlines()
    
    for i, list in enumerate(get_divided_list(json_list)):
        write_list(list, LABEL_PATH[i])
