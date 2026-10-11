import numpy as np
import json
import copy


class TriException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
    
    def __str__(self) -> str:
        return super().__str__()

class Tri:
    def __init__(self, a, b, c):
        self.TOLERANCE = -1e-9
        self.A_TOLERANCE = 1e-4
        self.a = a
        self.b = b
        self.c = c
    
    def __add__(self, other):
        return Tri(
            self.a + other.a,
            self.b + other.b,
            self.c + other.c
        )
    
    def __sub__(self, other):
        return Tri(
            self.a - other.a,
            self.b - other.b,
            self.c - other.c
        )

    def __mul__(self, other):
        """(ax + b)(cx + d)"""
        if isinstance(other, Tri) and other.a > self.TOLERANCE and self.a > self.TOLERANCE:
            return Tri(
                self.b * other.b,
                self.b * other.c + self.c * other.b,
                self.c * other.c
            )
        elif "int" in str(type(other)) or "float" in str(type(other)):
            return Tri(
                self.a * other,
                self.b * other,
                self.c * other
            )
        else:
            raise TriException("It is used for the multiplication of first-degree polynomials")
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, deg):
        if deg == 2:
            return self * self
        else:
            raise TriException("Only degree 2 is allowed")

    def __repr__(self) -> str:
        return f"{self.a}x^2 + {self.b}x + {self.c}"

    def solve(self):
        delta = self.b ** 2 - 4 * self.a * self.c

        if delta < 0:
            raise TriException("no solution")
        else:
            print("a", self.a)
            if self.a < self.A_TOLERANCE:
                return self.c / -self.b
            else:
                return (
                    (-self.b + np.sqrt(delta)) / (2 * self.a),
                    (-self.b - np.sqrt(delta)) / (2 * self.a)
                )
    
    def caculate(self, x):
        return self.a * x**2 + self.b * x + self.c

class VanishingPoint():
    def __init__(self, lanes, image_resolution=(720, 1280)) -> None:
        self.lanes = np.array(lanes)
        self.DELTA_Y = -10
        self.H, self.W = image_resolution
    
    def from_vp_get_point(self, vanishing_point, point):
        vp_x = vanishing_point[0]
        vp_y = vanishing_point[1]
        point_x = point[0]
        point_y = point[1]

        return (point_y - vp_y + self.DELTA_Y) * (vp_x - point_x) / (vp_y - point_y + 1e-5) + vp_x

    def filter_vp_x(self, x, min, max):
        if isinstance(x, tuple):
            x = [x_ for x_ in x if min < x_ < max]
            x = x[0] if x else max // 2
        
        return x

    def get_optimal_vp(self, starting_point, label, starting_y):
        temp_param = self.DELTA_Y / (label - starting_point).sum()
        constant = -temp_param * starting_point.sum() + starting_y
        vp_y = Tri(0, temp_param * len(label), constant) # from the partial derivative of vp_x
        
        temp_1 = temp_2 = Tri(0, 0, 0)
        for i in range(len(starting_point)):
            temp_1 += Tri(0, 1, -starting_point[i]) * (label[i] - starting_point[i])
            temp_2 += Tri(0, 1, -starting_point[i]) ** 2
        equation = (vp_y - Tri(0, 0, starting_y)) * temp_1 - self.DELTA_Y * temp_2 # from the partial derivative of vp_y
        vp_x = self.filter_vp_x(equation.solve(), 0, self.W) # degenerate into a linear equation‌

        return (vp_x, vp_y.caculate(vp_x))

    def gt_generator(self):
        for i in range(len(self.lanes[0]) - 1, -1, -1):
            gt_lanes = self.lanes[:, i]
            yield gt_lanes
    
    def get_vp_label(self):
        gt_gen = self.gt_generator()
        starting_point = self.lanes[:, -1]
        fit_vp = []
        tba_index = []

        for i in range(len(self.lanes[0]) - 1):
            gt_lanes = next(gt_gen)
            current_y = self.H - 10 - 10 * i
            non_zero_index = (starting_point > 0) & (gt_lanes > 0)
            if np.count_nonzero(non_zero_index) <= 1:
                fit_vp.append([self.W // 2, 240])
                starting_point = self.lanes[:, -i-1]
                continue
            
            vp = self.get_optimal_vp(starting_point[non_zero_index], gt_lanes[non_zero_index], current_y)
            fit_vp.append(vp)
            print("starting_point", starting_point[non_zero_index], "ground truth", gt_lanes[non_zero_index])
            print("vp", vp)

            starting_point = [self.from_vp_get_point(vp, (p, current_y)) for p in starting_point[non_zero_index]]
            temp_points = copy.copy(gt_lanes)
            temp_points[non_zero_index] = np.array(starting_point)
            starting_point = temp_points
            ...

            
if __name__ == "__main__":
    with open(r".\dataset\tusimple\train_vp.json", "r") as f:
        label_lines = f.readlines()

    for label in label_lines:
        lanes = json.loads(label)["lanes"] 
        vp = VanishingPoint(lanes)
        vp.get_vp_label()
        exit(0)
        