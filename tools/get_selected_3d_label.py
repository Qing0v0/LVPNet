import json
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.append(os.getcwd())
from utils.misc import map_2D_to_3D, fill_lane


PATH = r"E:\dataset\selected-ONCE-3dlanes\train"
IMAGE_PATH = r"E:\dataset\ONCE-3dlanes\data"
json_file = open("depth_label/depth_train.json", "r")

for line in json_file.readlines():
    label = json.loads(line)

    # 复制标签
    save_path = PATH + label["image_name"].replace("jpg", "json")
    calibration = np.array(label["calibration"])
    lanes = np.array(label["lanes"])
    depths = np.array(label["depth"])

    lanes_3d = []
    for i in range(len(lanes)):
        z = depths[i]
        lanes_3d.append(map_2D_to_3D(fill_lane(lanes[i], True), z[z > 0], calibration[0:3, 0:3]).tolist())
    
    save_info = {
        "lanes_num":len(lanes_3d),
        "lanes":lanes_3d,
        "calibration":calibration.tolist()
    }

    path = Path(save_path)
    if not path.parent.exists():
        os.makedirs(str(path.parent))
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(save_info) + '\n')
    
    # 复制图片
    # image_name = label["image_name"]
    # image = Image.open(IMAGE_PATH + image_name)

    # save_path = PATH + label["image_name"]
    # path = Path(save_path)
    # if not path.parent.exists():
    #     os.makedirs(str(path.parent))
    # image.save(save_path)

    # 生成列表
    # image_name = label["image_name"]
    # with open(r"E:\dataset\selected-ONCE-3dlanes\list\test.txt", "a") as f:
    #     f.writelines(image_name + "\n")