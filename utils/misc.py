# 存放一些多次用到的功能函数
import cv2
import numpy as np
from scipy.interpolate import interp1d


# 预定义可视化车道线颜色(蓝0 绿1 红2 青3 粉4 黄5 深绿6)
COLOR = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 128, 36),
    (35, 190, 64)
]

AVERAGE_DEPTH = np.array([
    33.90011111, 36.16625, 39.30155556, 42.54513554, 35.54903031, 28.79807681, 23.8762293, 
    20.37109063, 17.77033445, 15.7415747, 14.20234833, 12.90714798, 11.81120757, 10.91570429, 
    10.12930333, 9.4433413, 8.84219709, 8.30783395, 7.83468594, 7.41362325, 7.0358659, 6.69740248, 
    6.39110027, 6.11296295, 5.85694083, 5.62170341, 5.40420827, 5.20157784, 5.01606084, 4.84414563, 
    4.68351908, 4.5302479, 4.3883768, 4.25366831, 4.12650138, 4.007863, 3.89590781, 3.79020785, 
    3.69016222, 3.59529474, 3.50466014, 3.41935601, 3.33700963, 3.25908222, 3.18431338, 3.11350997, 
    3.04585924, 2.98100227, 2.91921386, 2.85987674
])

GET_Y = lambda x: x*10 +529
GET_INDEX = lambda x: int((x - 529) // 10)

# 对车道线标签进行插值
def interpolate_3d_tool(lanes):
    """
    each lane of input lanes must have at least two points
    input: lanes: [lane_num, ]
    """

    def cubic_interpolate(lane_spec):
        lane_spec = np.array(lane_spec)
        x = lane_spec[:, 0]
        y = lane_spec[:, 1]
        z = lane_spec[:, 2]
        z_min, z_max = min(z), max(z)
        zmin, zmax = np.ceil(z_min * 1000) / 1000, np.floor(z_max * 1000) / 1000
        gt_lane_points_camera = []
        # less than four points cannot do cubic spline interpolate, use linear instead
        # 步长为(zmax - zmin) / 6, 插值可以获得7个点
        if len(lane_spec) < 4:
            f_zx = interp1d(z, x, kind='linear', fill_value='extrapolate')
            f_zy = interp1d(z, y, kind='linear', fill_value='extrapolate')
            for z in np.arange(zmin, zmax, np.around((zmax - zmin) / 6, 3)):
                z = np.around(z, 3)
                x_out = np.around(f_zx(z), 3)
                y_out = np.around(f_zy(z), 3)
                gt_lane_points_camera.append([x_out, y_out, z])
        else:
            f_zx = interp1d(z, x, kind='cubic', fill_value='extrapolate')
            f_zy = interp1d(z, y, kind='cubic', fill_value='extrapolate')
            for z in np.arange(zmin, zmin + np.around((zmax - zmin) / 2, 1), 0.02):
                z = np.around(z, 3)
                x_out = np.around(f_zx(z), 3)
                y_out = np.around(f_zy(z), 3)
                gt_lane_points_camera.append([x_out, y_out, z])
            for z in np.arange(zmin + np.around((zmax - zmin) / 2, 1), zmax, 0.5):
                z = np.around(z, 3)
                x_out = np.around(f_zx(z), 3)
                y_out = np.around(f_zy(z), 3)

                gt_lane_points_camera.append([x_out, y_out, z])
        return gt_lane_points_camera

    lanes_inter = []
    for lane_spec in lanes:
        if len(lane_spec) < 2:
            continue
        lane_spec_inter = cubic_interpolate(lane_spec)
        lanes_inter.append(lane_spec_inter)
    return lanes_inter

# 从3D车道线映射到图像上
def get_2D_points_from_3D(lane, calibration):
    """
    @return:
    - lane_points: np.array[2 x N]  [(x1, y1), ...]
    """
    lane_points_3D = np.transpose(np.array(lane))
    calibration = calibration[:, 0:3]
    lane_points = calibration @ lane_points_3D / lane_points_3D[2, :]
    lane_points = lane_points[0:2, :]

    return lane_points

# 可视化
def draw_lanes(image, lanes, calibration, color_remove=None, interpolate=True):
    """
    通过3D的车道线标签在2D图像上画出车道线
    @params
    - image: np.array[H x W x 3] 图像
    - lanes: np.array[lane_num, ] 3D车道线的点[(x1, y1, z1), ...]
    - calibration: 内参矩阵
    - color_remove: (B, G, R), 车道线显示的颜色, 方便与为移除之前做对比
    - interpolate: 是否进行插值, 默认使用
    """
    if interpolate:
        lanes_interpolate = interpolate_3d_tool(lanes)
    else:
        lanes_interpolate = lanes

    for i in range(len(lanes_interpolate)):
        lane_points = get_2D_points_from_3D(lanes_interpolate[i], calibration)
        for j in range(lane_points.shape[1]):
            if i < len(COLOR):
                lane_color = color_remove if color_remove else COLOR[i]
            else:
                lane_color = (0, 0, 0)
            cv2.circle(image, (round(lane_points[0][j]), round(lane_points[1][j])), 5, lane_color, -1)

    return image

# 对预测的2D深度进行插值
def interpolate_for_2d_z(z:list, y:list):
    Z_STEP = 0.001
    def f(a, b, c, d, x):
        return a * x**2 + b * x + c + d / x

    def query_approximation(y, query_table, z_table):
        index = np.argmin(np.abs(query_table - y))
        return z_table[index]

    z, y = np.array(z), np.array(y)
    # 使用最小二乘法求出系数
    z_square = z * z
    z_reciprocal = 1 / z
    m = np.array([z_square, z, np.ones_like(z), z_reciprocal]).T
    pseudo_inverse = np.linalg.pinv(m.T @ m)
    parameters = pseudo_inverse @ (m.T @ y)
    
    # 获取查询表格, 进行快速非精确求解
    z_table = np.arange(z[0], z[-1], -Z_STEP)
    query_table = np.array([f(*parameters, z_) for z_ in z_table])

    # 进行插值
    z_interpolation = []
    start, end = y[:-1], y[1:]
    for i in range(len(start)):
        z_interpolation.append(z[i])
        for y_ in np.arange(start[i] + 1, end[i], 1):
            z_interpolation.append(query_approximation(y_, query_table, z_table))

    return z_interpolation

def interpolate_for_2d(depth, lanes):
    """
    对2D散点进行插值, 方便在图像上连续显示
    @params:
    - depth: [lane_num, :]
    - lanes: [lane_num, :]
    @return:
    - depth_interpolation: np.array([lanes_num, :])
    - lanes_interpolation: np.array([lanes_num, :, 2])
    """
    depth, lanes = np.array(depth), np.array(lanes)
    depth_interpolation = []
    lanes_interpolate = []

    for i in range(len(depth)):
        valid_index = np.where(lanes[i] > 0)[0]
        start_y, end_y = GET_Y(valid_index[0]), GET_Y(valid_index[-1])

        # 获取深度插值
        depth_interpolation.append(interpolate_for_2d_z(depth[i][valid_index], np.arange(start_y, end_y + 10, 10)))
        # 获取2D车道线插值
        lane = lanes[i][valid_index]
        f_lane = interp1d(np.arange(start_y, end_y + 10, 10), lane)
        lane_interpolate = []
        for y in range(start_y, end_y + 1):
            lane_interpolate.append([int(f_lane(y)), y])
        lanes_interpolate.append(lane_interpolate)

    return depth_interpolation, lanes_interpolate

# 变换车道线表达的形式[50] 或者 [N, 2]
def fill_lane(lane, unfill=False):
    if not unfill:
        filled_lane = np.zeros((50)) - 2
        for point in lane:
            filled_lane[GET_INDEX(point[1])] = point[0]
        return np.array(filled_lane)
    else:
        unfilled_lane = []
        for i in range(len(lane)):
            if lane[i] < 0:
                continue
            unfilled_lane.append([lane[i], GET_Y(i)])
        return np.array(unfilled_lane)

# 将2D车道线转换成3D车道线
def map_2D_to_3D(lane, z, calibration):
    inv_matrix = np.linalg.inv(calibration)

    if len(np.shape(z)) == 1:
        z = np.reshape(z, (-1, 1))
    lane = np.hstack([lane, np.ones((lane.shape[0], 1))])
    return np.transpose(inv_matrix @ np.transpose((lane * z)))