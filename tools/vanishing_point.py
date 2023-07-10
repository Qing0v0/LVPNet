from itertools import combinations

import numpy as np
from scipy.signal import savgol_filter


class VanishingPoint():
    def __init__(self) -> None:
        pass

    '''
    已知左车道两点、右车道两点,它们交点即为消失点
    input: left_lane_point   P1(x1, y1)  P2(x2, y1 - delta_y)
           right_lane_point  P3(x3, y1)  P4(x4, y1 - delta_y)
           delta_y: 10
    return: 左右车道交点(vanishing_point_x, vanishing_point_y)
    '''
    @staticmethod
    def get_vanishing_point(x1, x2, x3, x4, y1):
        vanishing_point_x = (x1*x4 - x2*x3)/(x4 - x3 - x2 + x1 + 1e-6)
        vanishing_point_y = 10 * (vanishing_point_x - x1) / (x1 - x2 + 1e-6) + y1

        return vanishing_point_x, vanishing_point_y

    '''
    已知消失点vanishing_point, 和当前点point, 获取车道线下一个点
    input: vanishing_point(x_vp, y_vp)
           point (x, y)
           delta_y: 车道线每个点之间的间隔
    return 车道线下一个点的x坐标
    '''
    @staticmethod
    def from_vp_get_point(vanishing_point, point, delta_y = -10):
        vp_x = vanishing_point[0]
        vp_y = vanishing_point[1]
        point_x = point[0]
        point_y = point[1]

        return (point_y - vp_y + delta_y) * (vp_x - point_x) / (vp_y - point_y + 1e-5) + vp_x

    '''
    通过已知的车道线标注,生成所有消失点
    input: lanes  N x point_nums, N表示最有有多少条车道线, point_nums表示每条车道线有多少点
    return: 消失点,
    '''
    @classmethod
    def from_lane_get_all_vp(cls, lanes, h_samples):
        lane_combinations = list(combinations(range(0, len(lanes)), 2))
        vp = []
        for i in range(len(lane_combinations)):
            left_lane = lanes[lane_combinations[i][0]]
            right_lane = lanes[lane_combinations[i][1]]
            lane_fiter = list(filter(lambda x: x[0]>0 and x[1]>0, zip(left_lane, right_lane, h_samples)))
            if(lane_fiter != []):
                left_lane, right_lane, y = zip(*lane_fiter)
            else:
                continue
            # 获取消失点
            for j in range(len(left_lane) - 1):
                vp.append(cls.get_vanishing_point(left_lane[j+1], left_lane[j], right_lane[j+1], right_lane[j], y[j+1]))

        # if(len(lanes) == 2):
        #     return vp, y
        return vp

    '''
    通过一系列消失点获取地平线位置
    input: 一系列消失点 N x 2
    return 1.消失点数目
           2.最佳内点数目
           3.地平线的y坐标
    '''
    @staticmethod
    def get_horizon(vanishing_points):
        vanishing_points = list(filter(lambda x: x[0]>0 and x[1] >0, vanishing_points))

        for sigma in range(3, 10):
            best_horizon_pos = 0
            best_inner_num = 0

            for i in range(500):
                inner_point = list(filter(lambda x: abs(x[1] - i) < sigma, vanishing_points))
                if(len(inner_point) > best_inner_num):
                    best_inner_num = len(inner_point)
                    best_horizon_pos = i
            # 内点数过半说明为地平线         
            if(best_inner_num / (len(vanishing_points)+1e-5) > 0.5):
                break

        return len(vanishing_points), best_inner_num, best_horizon_pos

    '''
    '''
    @classmethod
    def reconstruct_lane(cls, vp, start_point, start = 159, points_num = 56, val = False):
        lane = [start_point]
        for i in range(len(vp)):
            next_point = cls.from_vp_get_point(vp[i], lane[-1])
            lane.append([next_point, lane[-1][1]-10])

        fill_lane = np.zeros(points_num) - 2
        for i in range(len(lane)):
            fill_lane[int((lane[i][1]-start)/10)] = lane[i][0]

        if(val):
            return fill_lane
        return lane

    '''
    获取车道线消失点标签
    input: lanes 车道线标注点
           h_samples 采样点的y轴坐标
           horizon_pos 地平线y轴坐标
    return 消失点坐标   1+47
    '''
    @classmethod
    def get_vp_label(cls, lanes, h_samples, horizon_pos, W=1280):
        vp_label = np.zeros_like(h_samples, dtype = float)
        vp_label[0] = horizon_pos

        lanes = np.array(lanes)
        lane_num, point_num = np.shape(lanes)
        fit_lanes = np.zeros((lane_num, point_num)) - 2
        for i in range(point_num - 1):
            current_points = list(filter(lambda x : x>0, lanes[:, -i - 1]))
            if(len(current_points) == 0):
                vp_label[i + 1] = W // 2
                fit_lanes[:, -i - 1] = lanes[:, -i - 1]
            elif(len(current_points) == 1):
                for j in range(len(lanes[:, -i - 1])):
                    if(fit_lanes[j, -i - 1] <= 0):
                        fit_lanes[j, -i - 1] = lanes[j, -i - 1]
                index = list(lanes[:, -i - 1]).index(current_points[0])
                if(lanes[index, -i - 2] > 0):
                    vp_label[i + 1] = cls.from_vp_get_point([lanes[index, -i - 2], h_samples[-i - 2]],\
                                        [fit_lanes[index, -i - 1], h_samples[-i - 1]], horizon_pos - h_samples[-i - 1])
                else:
                    vp_label[i + 1] = W // 2
            else:
                for j in range(len(lanes[:, -i - 1])):
                    if(fit_lanes[j, -i - 1] <= 0):
                        fit_lanes[j, -i - 1] = lanes[j, -i - 1]

                min_error = 10000
                min_vp_x = -1
                for vp_x in range(0, W - 1):
                    point = np.zeros((lane_num))
                    for lane in range(len(fit_lanes[:, -i - 1])):
                        if(fit_lanes[lane, -i - 1] > 0):
                            point[lane] = cls.from_vp_get_point([vp_x, horizon_pos], [fit_lanes[lane, -i - 1], h_samples[-i - 1]])
                        else:
                            continue

                    error = 0
                    for j in range(lanes.shape[0]):
                        if(lanes[j, -i - 2] > 0):
                            error += abs(point[j] - lanes[j, -i - 2])

                    if(error < min_error):
                        min_vp_x = vp_x
                        min_error = error
                        fit_lanes[:, -i - 2] = point

                vp_label[i + 1] = min_vp_x

        return vp_label, fit_lanes
    
    @staticmethod
    def smoothe_vp(vp):
        vp[1:][vp[1:] != 640] = savgol_filter(vp[1:][vp[1:] != 640], 15, 3, mode= 'nearest')

        return vp

