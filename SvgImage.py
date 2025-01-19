import shapely.geometry

from main import *
import numpy as np
import drawsvg as draw
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry.linestring import LineString
from shapely import affinity


class SvgImage:
    def __init__(self, size, border):
        self.size = np.array([size[0], size[1]])
        self.lines = []
        self.border = Polygon(border)

    def refit_size(self):
        min_point = [1000000, 1000000]
        max_point = [0, 0]
        for line in self.lines:
            min_point[0] = min(min_point[0], line[0][0], line[1][0])
            min_point[1] = min(min_point[1], line[0][1], line[1][1])
            max_point[0] = max(max_point[0], line[0][0], line[1][0])
            max_point[1] = max(max_point[1], line[0][1], line[1][1])
        for i in range(len(self.lines)):
            self.lines[i][0] -= np.array(min_point)
            self.lines[i][1] -= np.array(min_point)
        self.size = np.array(max_point) - np.array(min_point)

    def add_line(self, start, end, overlap = True):
        d = end - start
        # joins duplicate lines
        for line in self.lines:
            d1 = line[1] - line[0]
            perpendicular = np.arctan2(d1[1], d1[0]) == np.arctan2(d[1], d[0])
            if perpendicular and (point_on_line(start, line) or point_on_line(end, line)):
                points = sorted([start, end, line[0], line[1]], key=lambda x: x[0])
                self.lines = list(filter(lambda a: a[0][0] != line[0][0] or a[1][0] != line[1][0] or
                                         a[0][1] != line[0][1] or a[1][1] != line[1][1], self.lines))
                self.lines.append([points[0].copy(), points[-1].copy()])
                return

        if not overlap:
            if self.border.contains(Point(start)):
                if self.border.contains(Point(end)):
                    return
                else:
                    shapely_line = LineString([start, end])
                    intersection = self.border.intersection(shapely_line)
                    intersections = []
                    if isinstance(intersection, shapely.geometry.base.BaseMultipartGeometry):
                        for i in intersection.geoms:
                            print(i)
                            intersections.extend(list(i.coords))
                    else:
                        intersections = list(intersection.coords)
                    if len(intersections) > 0:
                        if np.linalg.norm(intersections[0] - end) > 0.00001:
                            self.lines.append([np.array(intersections[0]), end.copy()])
                    return
            elif self.border.contains(Point(end)):
                shapely_line = LineString([start, end])
                intersection = self.border.intersection(shapely_line)
                intersections = []
                if isinstance(intersection, shapely.geometry.base.BaseMultipartGeometry):
                    for i in intersection.geoms:
                        print(i)
                        intersections.extend(list(i.coords))
                else:
                    intersections = list(intersection.coords)
                if len(intersections) > 0:
                    if np.linalg.norm(intersections[0] - start) > 0.00001:
                        self.lines.append([start.copy(), np.array(intersections[0])])
                return

        self.lines.append([start.copy(), end.copy()])

    def translate(self, translation):
        for i in range(len(self.lines)):
            self.lines[i][0] += translation
            self.lines[i][1] += translation
        self.border = affinity.translate(self.border, translation[0], translation[1])

    def rotate(self, angle, anchor=np.array([0, 0])):
        for i in range(len(self.lines)):
            self.lines[i][0] = rotate_vector(self.lines[i][0] - anchor, angle) + anchor
            self.lines[i][1] = rotate_vector(self.lines[i][1] - anchor, angle) + anchor
        self.border = affinity.rotate(self.border, angle, origin=(anchor[0], anchor[1]), use_radians=True)

    def mirror(self):
        inside_lines = []
        for i in range(len(self.lines)):
            s = self.lines[i][0].copy()
            e = self.lines[i][1].copy()
            s[0] = self.size[0] - s[0]
            e[0] = self.size[0] - e[0]
            inside_lines.append([s, e])

        output = SvgImage(self.size, [])
        output.border = affinity.scale(self.border, xfact=-1, yfact=1, zfact=1, origin=(self.size[0] / 2, self.size[1] / 2))
        output.lines = inside_lines
        return output

    def impose(self, other, overlay=True):
        for line in other.lines:
            self.add_line(line[0], line[1], overlay)
        try:
            self.border = self.border.union(other.border)
        except:
            print(self.border)
            print(other.border)

    def cut(self, polygon):
        inside_lines = []
        p = Polygon(polygon)
        for line in self.lines:
            if p.contains(Point(line[0])):
                if p.contains(Point(line[1])):
                    inside_lines.append([line[0].copy(), line[1].copy()])
                else:
                    shapely_line = LineString([line[0], line[1]])

                    intersection = p.intersection(shapely_line)
                    intersections = []
                    if isinstance(intersection, shapely.geometry.base.BaseMultipartGeometry):
                        for i in intersection.geoms:
                            print(i)
                            intersections.extend(list(i.coords))
                    else:
                        intersections = list(intersection.coords)

                    if len(intersections) > 0:
                        if np.linalg.norm(intersections[-1] - line[0]) > 0.00001:
                            inside_lines.append([line[0].copy(), np.array(intersections[-1])])
            elif p.contains(Point(line[1])):
                shapely_line = LineString([line[0], line[1]])

                intersection = p.intersection(shapely_line)
                intersections = []
                if isinstance(intersection, shapely.geometry.base.BaseMultipartGeometry):
                    for i in intersection.geoms:
                        print(i)
                        intersections.extend(list(i.coords))
                else:
                    intersections = list(intersection.coords)

                if len(intersections) > 0:
                    if np.linalg.norm(intersections[0] - line[1]) > 0.00001:
                        inside_lines.append([np.array(intersections[0]), line[1].copy()])
            else:
                shapely_line = LineString([line[0], line[1]])

                intersection = p.intersection(shapely_line)
                intersections = []
                if isinstance(intersection, shapely.geometry.base.BaseMultipartGeometry):
                    for i in intersection.geoms:
                        print(i)
                        intersections.extend(list(i.coords))
                else:
                    intersections = list(intersection.coords)

                if len(intersections) >= 2:
                    inside_lines.append([np.array(intersections[0]), np.array(intersections[-1])])
        output = SvgImage(self.size, polygon)
        output.lines = inside_lines
        for i in range(len(polygon) - 1):
            output.add_line(polygon[i], polygon[i + 1])
        return output

    def draw_image(self, line_width):
        image = draw.Drawing(self.size[0], self.size[1], origin='top-left')
        for line in self.lines:
            image.append(draw.Line(line[0][0], line[0][1], line[1][0], line[1][1], stroke='black', stroke_width=line_width))
        return image


def point_on_line(point, line):
    d_ = point - line[0]
    d = line[1] - line[0]
    if d_[0] == d[0] == 0:
        if d_[1] == d[1] == 0:
            print(line)
            print("WTFkansfoiasbhfoasbfowdbhfowho")
            return True
        elif d[1] != 0:
            return 0 <= d_[1] / d[1] <= 1
        else:
            return False
    elif d[0] != 0:
        if d_[1] == d[1] == 0:
            return 0 <= d_[0] / d[0] <= 1
        elif d[1] != 0:
            d_x = d_[0] / d[0]
            d_y = d_[1] / d[1]
            return d_x == d_y and 0 <= d_x <= 1
        else:
            return False
    else:
        return False


def rotate_vector(vector, theta):
    return np.dot(np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]), vector)