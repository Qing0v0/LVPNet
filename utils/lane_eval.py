import numpy as np
import cv2
from munkres import Munkres
from shapely.geometry import LineString


class Eval:
    def __init__(self, pred_lanes, gt_lanes) -> None:
        # config
        self.side_range_l = -10
        self.side_range_h = 10
        self.fwd_range_l = 0
        self.fwd_range_h = 50
        self.height_range_l = 0
        self.height_range_h = 5
        self.res = 0.05
        self.lane_width_x = 30
        self.lane_width_y = 10

        # bev project config
        self.zx_xmax = int((self.side_range_h - self.side_range_l) / self.res)
        self.zx_ymax = int((self.fwd_range_h - self.fwd_range_l) / self.res)
        self.zy_xmax = self.zx_ymax
        self.zy_ymax = int((self.height_range_h - self.height_range_l) / self.res)
        
        self.iou_thresh = 0.3
        self.distance_thresh = 0.3
        self.pred_lanes = pred_lanes
        self.gt_lanes = gt_lanes
    
    @classmethod
    def sort_lanes_z(self, lanes):
        sorted_lanes = list()
        for lane_spec in lanes:
            if lane_spec[0][-1] > lane_spec[1][-1]:
                lane_spec = lane_spec[::-1]
            sorted_lanes.append(lane_spec)
        return sorted_lanes
    
    # BEV Project
    def proj_oneline_zx(self, one_lane):
        img = np.zeros([self.zx_ymax, self.zx_xmax], dtype=np.uint8)

        one_lane = np.array(one_lane)
        one_lane = one_lane[one_lane[:, 2] < 10]
        lane_x = one_lane[:, 0]
        lane_z = one_lane[:, 2]

        x_img = (lane_x / self.res).astype(np.int32)
        y_img = (-lane_z / self.res).astype(np.int32)

        x_img += int(self.side_range_h / self.res)
        y_img += int(self.fwd_range_h / self.res)

        # img[y_img, x_img] = 255
        for i in range(y_img.shape[0]-1):
            cv2.line(img, (x_img[i], y_img[i]), (x_img[i+1], y_img[i+1]), 255, self.lane_width_x)
        return img

    def cal_mean_dist(self, src_line, dst_line):
        """
        :param src_line: gt
        :param dst_line: pred
        :return:
        """
        src_line = LineString(np.array(src_line))
        dst_line = LineString(np.array(dst_line))

        total_distance = 0
        samples = np.arange(0.05, 1, 0.1)
        for sample in samples:
            total_distance += src_line.interpolate(sample, normalized=True).distance(dst_line)
        mean_distance = total_distance / samples.shape[0]
        return mean_distance
    
    def calc_iou(self, lane1, lane2):
        """
        :param lane1:
        :param lane2:
        :return:
        """
        img1, img2 = self.proj_oneline_zx(lane1), self.proj_oneline_zx(lane2)
        # cv2.imshow("img1", cv2.resize(img1, None, fx=1, fy=0.5))
        # cv2.imshow("img2", cv2.resize(img2, None, fx=1, fy=0.5))
        # cv2.waitKey(0)

        union_im = cv2.bitwise_or(img1, img2)
        union_sum = union_im.sum()
        inter_sum = img1.sum() + img2.sum() - union_sum
        if union_sum == 0:
            return 0
        else:
            return inter_sum / float(union_sum)
    
    def eval(self):
        self.gt_lanes, self.pred_lanes = self.sort_lanes_z(self.gt_lanes), self.sort_lanes_z(self.pred_lanes)
        tp, distance_error = self.cal_tp(self.gt_lanes, self.pred_lanes)

        return tp, distance_error

    def cal_tp(self, gt_lanes, pred_lanes):
        tp = 0
        distance_error = 0
        gt_num = len(gt_lanes)
        pred_num = len(pred_lanes)

        if gt_num != 0 and pred_num != 0:
            iou_mat = [[0 for col in range(pred_num)] for row in range(gt_num)]
            for i in range(gt_num):
                for j in range(pred_num):
                    iou_mat[i][j] = self.calc_iou(gt_lanes[i], pred_lanes[j])
            cost_mat = []
            for row in iou_mat:
                cost_row = list()
                for col in row:
                    cost_row.append(1.0 - col)
                cost_mat.append(cost_row)
            m = Munkres()
            match_idx = m.compute(cost_mat)  #
            for row, col in match_idx:
                gt_lane = gt_lanes[row]
                pred_lane = pred_lanes[col]
                cur_distance = self.cal_mean_dist(gt_lane, pred_lane)
                if cur_distance < self.distance_thresh:
                    distance_error += cur_distance
                    tp += 1

        return tp, distance_error

