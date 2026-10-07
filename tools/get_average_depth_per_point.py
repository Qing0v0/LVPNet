# 获取每个垂直位置点的平均深度
import json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


with open(r"depth_label/depth_train.json", "r", encoding="utf-8") as f:
    lanes_label = f.readlines()
    depth = []
    for lane in lanes_label:
        lane = json.loads(lane)
        for d in lane["depth"]:
            depth.append(d)

depth = np.array(depth)
print(f"最大深度为{np.max(depth)}")


def eigenlanes_like(depth):
    U, S, V = np.linalg.svd(depth.T)

    max_singular_value = np.diag(S).sum()
    num_components = 15
    print(np.diag(S[:num_components]).sum() / max_singular_value)
    U_reduced = U[:, :num_components]
    C = U_reduced.T @ depth.T

    model = KMeans(n_clusters=1)
    model.fit(C.T)

    centers = model.cluster_centers_
    results = U_reduced @ centers.T
    relative_str = "["
    for r in results:
        relative_str += str(r[0])
        relative_str += ", "
    relative_str += "]"
    print(relative_str)

def kmeans(depth):
    model = KMeans(n_clusters=1)
    model.fit(depth)

    centers = model.cluster_centers_
    relative_str = "["
    for r in centers.T:
        relative_str += str(r[0])
        relative_str += ", "
    relative_str += "]"
    print(relative_str)

eigenlanes_like(depth)
# kmeans(depth)

# average_depth_per_point = np.zeros((50))
# for i in range(50):
#     i_th_point_depth = depth[:, i]
#     i_th_point_depth = i_th_point_depth[i_th_point_depth > 0]
#     average_depth_per_point[i] = np.mean(i_th_point_depth)

# print(f"每个点的平均深度为{average_depth_per_point}")