from vanishing_point import *
import json


def save_vp_label(lanes_label_lines, exist_lines, save_name):
    for index, line in enumerate(lanes_label_lines):
        data = json.loads(line) 
        lanes = data['lanes']
        h_samples = data['h_samples']
        file_name = data['raw_file']
        lane_exist = exist_lines[index].split(" ")[2:]
        lane_exist = list(map(int, lane_exist))

        vp = VanishingPoint.from_lane_get_all_vp(lanes, h_samples)
        _, _, pos = VanishingPoint.get_horizon(np.array(vp))
        vanishing_points, _ = VanishingPoint.get_vp_label(lanes, h_samples, pos)
        vanishing_points = VanishingPoint.smoothe_vp(vanishing_points)
        if(len(vanishing_points) == 48):
            vanishing_points = np.append(vanishing_points, np.zeros((8)))

        lanes = [lane for lane in lanes if (np.array(lane)>0).any()]
        
        lane_label = []
        j = 0
        for i in range(6):
            if(lane_exist[i] == 0):
                lane_label.append(list(np.zeros_like(lanes[0], dtype = float) - 2))
            else:
                lane_label.append(list(lanes[j]))
                j += 1
        
        vp_label = {"file_name":file_name, "vp_label":list(vanishing_points), "lanes":list(lane_label)}
        with open(save_name, 'a', encoding='utf-8') as file:
            file.writelines(json.dumps(vp_label) + '\n')

if __name__ == '__main__':
    train_set = ["/label_data_0313.json", "/label_data_0601.json"]
    val_set = ["/label_data_0531.json"]
    base_path = r"dataset\\tusimple"

    # get train vp label
    lanes_label_lines = []
    exist_lines = []
    for file_name in train_set:
        with open(base_path + file_name, 'r') as lanes_label_file:
            lanes_label_lines += lanes_label_file.readlines()
        with open(base_path + "\\lists\\list6_train.txt", 'r') as exist_file:
            exist_lines += exist_file.readlines()
    save_vp_label(lanes_label_lines, exist_lines, "list6_train.json")
    
    # get val vp label
    for file_name in val_set:
        with open(base_path + file_name, 'r') as lanes_label_file:
            lanes_label_lines = lanes_label_file.readlines()
        with open(base_path + "\\lists\\list6_val.txt", 'r') as exist_file:
            exist_lines = exist_file.readlines()
    save_vp_label(lanes_label_lines, exist_lines, "list6_val.json")
    