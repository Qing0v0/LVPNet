import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from matplotlib.ticker import FuncFormatter

colors = [[1, 0, 0],  # red
          [0, 1, 0],  # green
          [0, 0, 1],  # blue
          [1, 0, 1],  # purple
          [0, 1, 1],  # cyan
          [1, 0.7, 0],  # orange
          [0.45, 0.17, 0.07]]  # crown

def get_xyz(lane_point):
    lane_points = lane_point
    lane_points = np.array(lane_points)
    x = lane_points[:, 0]
    y = lane_points[:, 1]
    z = lane_points[:, 2]
    t = np.arange(1, lane_points.shape[0] + 1)  # simple assumption that data was sampled in regular steps
    fitx = np.polyfit(t, x, 3)                  # polyfit
    fity = np.polyfit(t, y, 2)
    fitz = np.polyfit(t, z, 2)
    lane = []
    for j in t:
        x_out = np.polyval(fitx, j)
        y_out = np.polyval(fity, j)
        z_out = np.polyval(fitz, j)
        lane.append([x_out, y_out, z_out])
    lane = np.array(lane)
    lane_x, lane_y, lane_z = lane[:, 0], lane[:, 1], lane[:, 2]

    return lane_x, lane_y, lane_z
    


def draw_lanes3d(all_lane_points, label):
    """
    visual all lane points in 3D ax plot, in the final plot, the coordinates have changed to normal coordinates
    input:
    all_lane_points: [lane_num, ]  all_lane_points[0]: [N,3]
    save_path: the dir path of saved image
    """
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    all_lane_points = all_lane_points
    for i in range(len(all_lane_points)):
        lane_x, lane_y, lane_z = get_xyz(all_lane_points[i])
        color = colors[0]
        ax.plot(lane_x, lane_z, -lane_y, color=color, linewidth=3, label='3D Lane %d' % i)  # normal coordinates
        # ax.scatter(lane_x, lane_z, -lane_y, s=1, c=np.array(color).reshape(1, -1))
    
    for i in range(len(label)):
        lane_x, lane_y, lane_z = get_xyz(label[i])
        color = colors[1]
        ax.plot(lane_x, lane_z, -lane_y, color=color, linewidth=3, label='label %d' % i) 

    def inverse_z(z, position):
        return "{:.3}".format(-z)

    ax.zaxis.set_major_formatter(FuncFormatter(inverse_z))
    ax.legend()
    plt.show()