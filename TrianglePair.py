from enum import Enum
#from PIL import Image, ImageDraw, ImageOps, ImageChops, ImageFont
from main import *
import numpy as np
import drawsvg as draw
from SvgImage import *

class Orientation(Enum):
    UNSET = 0
    LEFT = 1
    RIGHT = -1


def copy_paste_polygon(target, side, origin, polygon):
    a1 = side[1] - side[0]
    a2 = polygon[1] - polygon[0]
    angles = np.arctan2(a1[0] * a2[1] - a1[1] * a2[0], np.dot(a1, a2))

    li = [h for h in polygon]
    li.append(polygon[0])

    cutout = origin.cut(li)
    cutout.translate(side[0]-polygon[0])
    cutout.rotate(-angles, side[0])
    target.impose(cutout, overlay=False)

    return target


class TrianglePair:
    orientation = Orientation.UNSET

    def __init__(self, tri1, tri2, connectedtriangles1, connectedtriangles2):
        self.triangle1 = tri1
        self.triangle2 = tri2
        self.connectedtriangles1 = connectedtriangles1
        self.connectedtriangles2 = connectedtriangles2
        self.orientation = Orientation.UNSET

        self.image_back = None
        self.image_front = None

        tri1.pair = self
        tri2.pair = self

    def __str__(self):
        return f"{self.triangle1}, {self.triangle2}: {self.orientation}"

    def set_orientation(self, new_orientation):
        if self.orientation == Orientation.UNSET:
            self.orientation = new_orientation
            for i in self.connectedtriangles1:
                if i.C == self.triangle1.C:
                    i.set_orientation(new_orientation)
                elif i.C == self.triangle1.A or i.C == self.triangle1.B:
                    if new_orientation == Orientation.LEFT:
                        i.set_orientation(Orientation.RIGHT)
                    elif new_orientation == Orientation.RIGHT:
                        i.set_orientation(Orientation.LEFT)
                else:
                    print("fuck fuck fuck")

            for i in self.connectedtriangles2:
                if i.C == self.triangle2.C:
                    i.set_orientation(new_orientation)
                elif i.C == self.triangle2.A or i.C == self.triangle2.B:
                    if new_orientation == Orientation.LEFT:
                        i.set_orientation(Orientation.RIGHT)
                    elif new_orientation == Orientation.RIGHT:
                        i.set_orientation(Orientation.LEFT)
                else:
                    print("GOD FUCKING DAMMIT")
        elif self.orientation != new_orientation:
            print("FOCK FOFK", self.orientation.name, new_orientation.name)

    def draw_corougation(self):
        self.triangle1.drawCorrugation()
        self.triangle2.drawCorrugation()
        self.triangle1.draw_bottom_flap(self.triangle2)
        self.triangle2.draw_bottom_flap(self.triangle1)

    def draw_connections(self):
        for t in self.connectedtriangles1:
            if self.orientation == Orientation.RIGHT:
                if t.containEdge((self.triangle1.A, self.triangle1.C)):
                    self.triangle1.draw_connecting_flap(t)
            else:
                if t.containEdge((self.triangle1.B, self.triangle1.C)):
                    self.triangle1.draw_connecting_flap(t)

        for t in self.connectedtriangles2:
            if self.orientation == Orientation.RIGHT:
                if t.containEdge((self.triangle2.A, self.triangle2.C)):
                    self.triangle2.draw_connecting_flap(t)
            else:
                if t.containEdge((self.triangle2.B, self.triangle2.C)):
                    self.triangle2.draw_connecting_flap(t)

    def draw_image(self):
        '''t2 = self.triangle2.image_back.rotate(180)
        to2 = np.array(t2.size) - self.triangle2.vertex_b_image_pos[0]

        offset = self.triangle1.vertex_b_image_pos[1] - to2
        corners = np.array([(0, 0), self.triangle1.image_back.size, offset, offset + np.array(t2.size)])
        negative_min_corner = -corners.min(axis=0)
        max_corner = corners.max(axis=0)

        offset += negative_min_corner
        size = negative_min_corner + max_corner

        self.image_back = Image.new("L", (int(size[0]), int(size[1])), 255)
        self.image_back.paste(self.triangle1.image_back, (int(negative_min_corner[0]), int(negative_min_corner[1])))
        self.image_back.paste(t2, (int(offset[0]), int(offset[1])), mask=ImageChops.invert(t2))

        t2 = self.triangle2.image_front.rotate(180)
        to2 = np.array(t2.size) - self.triangle2.vertex_f_image_pos[0]

        offset = self.triangle1.vertex_f_image_pos[1] - to2
        corners = np.array([(0, 0), self.triangle1.image_front.size, offset, offset + np.array(t2.size)])
        negative_min_corner = -corners.min(axis=0)
        max_corner = corners.max(axis=0)

        offset += negative_min_corner
        size = negative_min_corner + max_corner

        self.image_front = Image.new("L", (int(size[0]), int(size[1])), 255)
        self.image_front.paste(self.triangle1.image_front, (int(negative_min_corner[0]), int(negative_min_corner[1])))
        self.image_front.paste(t2, (int(offset[0]), int(offset[1])), mask=ImageChops.invert(t2))

        self.image_back = ImageOps.mirror(self.image_back)'''

        self.image_front = SvgImage(np.array([0, 0]), [])
        self.image_front.impose(self.triangle1.image_front)
        self.image_front.rotate(np.deg2rad(180), self.triangle1.vertex_f_image_pos[0])
        self.image_front.translate(self.triangle2.vertex_f_image_pos[1] - self.triangle1.vertex_f_image_pos[0])
        self.image_front.impose(self.triangle2.image_front)

        self.image_back = SvgImage(np.array([0, 0]), [])
        self.image_back.impose(self.triangle1.image_back)
        self.image_back.rotate(np.deg2rad(180), self.triangle1.vertex_b_image_pos[0])
        self.image_back.translate(self.triangle2.vertex_b_image_pos[1] - self.triangle1.vertex_b_image_pos[0])
        self.image_back.impose(self.triangle2.image_back)

        self.image_back = self.image_back.mirror()

        #self.image_front.show()
        #self.image_back.show()


class Triangle:
    pair = None
    angles = [0, 0, 0]
    angles_max = [0, 0, 0]
    in_angle = 0

    image_front = None
    image_back = None
    # A, B, C1, G, C2, O, G'
    vertex_b_image_pos = [np.array([0.0, 0.0]) for i in range(7)]
    vertex_f_image_pos = [np.array([0.0, 0.0]) for i in range(7)]

    def __init__(self, a, b, c, A, B, C):
        self.id = 0
        self.a = a
        self.b = b
        self.c = c
        self.A = A
        self.B = B
        self.C = C

        self.pair = None

        self.angles = [np.arccos(-(a*a - b*b - c*c) / (b * c * 2)),
                       np.arccos(-(b*b - a*a - c*c) / (a * c * 2)),
                       np.arccos(-(c*c - b*b - a*a) / (b * a * 2))]

        self.angles_max = [0, 0, 0]
        self.in_angle = np.pi / 2

        self.vertex_b_image_pos = [np.array([0.0, 0.0]) for i in range(7)]

        self.image_front = None
        self.image_back = None

        self.vertex_b_image_pos = [np.array([0.0, 0.0]) for i in range(7)]
        self.vertex_f_image_pos = [np.array([0.0, 0.0]) for i in range(7)]

    def __str__(self):
        return f"[{self.A}, {self.B}, {self.C}]"

    def vertices(self):
        return (self.A, self.B, self.C)

    def containVertex(self, vertex):
        return vertex == self.A or vertex == self.B or vertex == self.C

    def edges(self):
        return [sorted([self.A, self.B]), sorted([self.B, self.C]), sorted([self.C, self.A])]

    def containEdge(self, edge):
        return edge[0] in [self.A, self.B, self.C] and edge[1] in [self.A, self.B, self.C]

    def adjacent(self, another):
        i = 0
        if another.A in [self.A, self.B, self.C]:
            i += 1
        if another.B in [self.A, self.B, self.C]:
            i += 1
        if another.C in [self.A, self.B, self.C]:
            i += 1
        return i == 2

    def adjacent_edge(self, another):
        x1 = self.edges()
        x2 = another.edges()
        x = [t for t in x1 if t in x2]
        x = x[0]
        return x

    def rotateClockwise(self):
        temp = self.A
        self.A = self.B
        self.B = self.C
        self.C = temp
        '''temp = self.a
        self.a = self.c
        self.c = self.b
        self.b = temp'''
        temp = self.a
        self.a = self.b
        self.b = self.c
        self.c = temp
        temp = self.angles[0]
        self.angles[0] = self.angles[1]
        self.angles[1] = self.angles[2]
        self.angles[2] = temp

    def rotateCounterClockwise(self):
        temp = self.A
        self.A = self.C
        self.C = self.B
        self.B = temp
        temp = self.a
        self.a = self.c
        self.c = self.b
        self.b = temp
        temp = self.angles[0]
        self.angles[0] = self.angles[2]
        self.angles[2] = self.angles[1]
        self.angles[1] = temp

    def set_orientation(self, orientation):
        self.pair.set_orientation(orientation)

    def drawImageBack(self):
        A = self.angles_max[0]/2
        B = self.angles_max[1]/2
        C = self.angles_max[2]/2
        self.vertex_b_image_pos[1] = self.vertex_b_image_pos[0] + np.array([self.c, 0]) * pixel_per_centimeter
        self.vertex_b_image_pos[2] = self.vertex_b_image_pos[1] + np.array(
            [-self.a * cos(B * 2), self.a * sin(B * 2)]) * pixel_per_centimeter
        self.vertex_b_image_pos[4] = self.vertex_b_image_pos[0] + np.array(
            [self.b * cos(A * 2), self.b * sin(A * 2)]) * pixel_per_centimeter
        d = self.c / (cos(B) * sin(A) / sin(B) + cos(A))
        self.vertex_b_image_pos[5] = self.vertex_b_image_pos[0] + np.array(
            [d * cos(A), d * sin(A)]) * pixel_per_centimeter

        f1 = np.linalg.norm(self.vertex_b_image_pos[4] - self.vertex_b_image_pos[5])
        f2 = np.linalg.norm(self.vertex_b_image_pos[2] - self.vertex_b_image_pos[5])

        open_angle = np.arccos(np.dot(self.vertex_b_image_pos[4] - self.vertex_b_image_pos[5],
                                      self.vertex_b_image_pos[2] - self.vertex_b_image_pos[5]) / f1 / f2) / 2

        if self.pair != None and self.pair.orientation == Orientation.RIGHT:
            self.vertex_b_image_pos[6] = self.vertex_b_image_pos[4] + \
                                         rotate_vector((self.vertex_b_image_pos[5] - self.vertex_b_image_pos[4]) / sin(
                                         C + open_angle) * sin(open_angle), -C)
        elif self.pair != None:
            self.vertex_b_image_pos[6] = self.vertex_b_image_pos[2] + \
                                     rotate_vector((self.vertex_b_image_pos[5] - self.vertex_b_image_pos[2]) / sin(
                                         C + open_angle) * sin(open_angle), C)

        self.vertex_b_image_pos[3] = self.vertex_b_image_pos[4] + \
                                     rotate_vector((self.vertex_b_image_pos[5] - self.vertex_b_image_pos[4]) / sin(
                                         C + open_angle) * sin(open_angle), C)

        m = np.array(self.vertex_b_image_pos).min(axis=0)
        for i in range(len(self.vertex_b_image_pos)):
            self.vertex_b_image_pos[i] -= m

        m = np.array(self.vertex_b_image_pos).max(axis=0)

        #self.image_back = Image.new("L", ((int)(m[0]), (int)(m[1])), 255)
        self.image_back = SvgImage((m[0], m[1]), [self.vertex_b_image_pos[0], self.vertex_b_image_pos[1], self.vertex_b_image_pos[2],
                                                  self.vertex_b_image_pos[3], self.vertex_b_image_pos[4], self.vertex_b_image_pos[0]])

        #draw = ImageDraw.Draw(self.image_back)

        distances = [np.linalg.norm(np.cross(self.vertex_b_image_pos[5] - self.vertex_b_image_pos[i],
                     self.vertex_b_image_pos[(i + 1) % 5] - self.vertex_b_image_pos[i])) /
                     np.linalg.norm(self.vertex_b_image_pos[(i + 1) % 5] - self.vertex_b_image_pos[i])
                     for i in range(5)]
        number_of_corougation = int(np.floor(min(distances) / (tolerance * pixel_per_centimeter)))

        for i in range(5):
            n1 = (self.vertex_b_image_pos[i] - self.vertex_b_image_pos[5]) / number_of_corougation
            n2 = (self.vertex_b_image_pos[(i + 1) % 5] - self.vertex_b_image_pos[5]) / number_of_corougation
            for t in range(1, number_of_corougation + 1):
                nn1 = n1 * t + self.vertex_b_image_pos[5]
                nn2 = n2 * t + self.vertex_b_image_pos[5]
                self.image_back.add_line(nn1, nn2)
            self.image_back.add_line(self.vertex_b_image_pos[i], self.vertex_b_image_pos[5])

        #self.image_back.append(draw.Text(self.id.__str__(), font_size=guide_font_size, x=self.vertex_b_image_pos[5][0], y=self.vertex_b_image_pos[5][1], text_anchor='middle'))

        #self.image_back.show()

    def drawImageFront(self):
        A = self.angles_max[0] / 2
        B = self.angles_max[1] / 2
        C = self.angles_max[2] / 2
        self.vertex_f_image_pos[1] = self.vertex_f_image_pos[0] + np.array([self.c, 0]) * pixel_per_centimeter
        if self.pair.orientation == Orientation.RIGHT:
            self.vertex_f_image_pos[4] = self.vertex_f_image_pos[0] + np.array(
                [self.b * cos(A * 2), self.b * sin(A * 2)]) * pixel_per_centimeter
            self.vertex_f_image_pos[3] = self.vertex_f_image_pos[4] + np.array(
                [self.a * cos(C * 2 - (np.pi - A * 2)), self.a * sin(C * 2 - (np.pi - A * 2))]) * pixel_per_centimeter
            d = self.c / (cos(B) * sin(A) / sin(B) + cos(A))
            self.vertex_f_image_pos[5] = self.vertex_f_image_pos[0] + np.array(
                [d * cos(A), d * sin(A)]) * pixel_per_centimeter

            f1 = np.linalg.norm(self.vertex_f_image_pos[1] - self.vertex_f_image_pos[5])
            f2 = np.linalg.norm(self.vertex_f_image_pos[3] - self.vertex_f_image_pos[5])

            open_angle = np.arccos(np.dot(self.vertex_f_image_pos[1] - self.vertex_f_image_pos[5],
                                          self.vertex_f_image_pos[3] - self.vertex_f_image_pos[5]) / f1 / f2) / 2

            self.vertex_f_image_pos[2] = self.vertex_f_image_pos[1] + \
                                         rotate_vector((self.vertex_f_image_pos[5] - self.vertex_f_image_pos[1]) / sin(
                                             B + open_angle) * sin(open_angle), -B)

            self.vertex_f_image_pos[6] = self.vertex_f_image_pos[1] + \
                                         rotate_vector((self.vertex_f_image_pos[5] - self.vertex_f_image_pos[1]) / sin(
                                             B + open_angle) * sin(open_angle), B)
        else:
            self.vertex_f_image_pos[2] = self.vertex_f_image_pos[1] + np.array(
                [-self.a * cos(B * 2), self.a * sin(B * 2)]) * pixel_per_centimeter
            self.vertex_f_image_pos[3] = self.vertex_f_image_pos[2] + np.array(
                [-self.b * cos(C * 2 - (np.pi - B * 2)), self.b * sin(C * 2 - (np.pi - B * 2))]) * pixel_per_centimeter
            d = self.c / (cos(B) * sin(A) / sin(B) + cos(A))
            self.vertex_f_image_pos[5] = self.vertex_f_image_pos[0] + np.array(
                [d * cos(A), d * sin(A)]) * pixel_per_centimeter

            f1 = np.linalg.norm(self.vertex_f_image_pos[0] - self.vertex_f_image_pos[5])
            f2 = np.linalg.norm(self.vertex_f_image_pos[3] - self.vertex_f_image_pos[5])

            open_angle = np.arccos(np.dot(self.vertex_f_image_pos[3] - self.vertex_f_image_pos[5],
                                          self.vertex_f_image_pos[0] - self.vertex_f_image_pos[5]) / f1 / f2) / 2

            self.vertex_f_image_pos[4] = self.vertex_f_image_pos[3] + \
                                         rotate_vector((self.vertex_f_image_pos[5] - self.vertex_f_image_pos[3]) / sin(
                                             A + open_angle) * sin(open_angle), -A)

            self.vertex_f_image_pos[6] = self.vertex_f_image_pos[3] + \
                                         rotate_vector((self.vertex_f_image_pos[5] - self.vertex_f_image_pos[3]) / sin(
                                             A + open_angle) * sin(open_angle), A)

        m = np.array(self.vertex_f_image_pos).min(axis=0)
        for i in range(len(self.vertex_f_image_pos)):
            self.vertex_f_image_pos[i] -= m

        m = np.array(self.vertex_f_image_pos).max(axis=0)

        self.image_front = SvgImage((m[0], m[1]), [self.vertex_f_image_pos[0], self.vertex_f_image_pos[1], self.vertex_f_image_pos[2],
                                                   self.vertex_f_image_pos[3], self.vertex_f_image_pos[4], self.vertex_f_image_pos[0]])

        distances = [np.linalg.norm(np.cross(self.vertex_f_image_pos[5] - self.vertex_f_image_pos[i],
                     self.vertex_f_image_pos[(i + 1) % 5] - self.vertex_f_image_pos[i])) /
                     np.linalg.norm(self.vertex_f_image_pos[(i + 1) % 5] - self.vertex_f_image_pos[i])
                     for i in range(5)]
        number_of_corougation = int(np.floor(min(distances) / (tolerance * pixel_per_centimeter)))

        for i in range(5):
            n1 = (self.vertex_f_image_pos[i] - self.vertex_f_image_pos[5]) / number_of_corougation
            n2 = (self.vertex_f_image_pos[(i + 1) % 5] - self.vertex_f_image_pos[5]) / number_of_corougation
            for t in range(1, number_of_corougation + 1):
                nn1 = n1 * t + self.vertex_f_image_pos[5]
                nn2 = n2 * t + self.vertex_f_image_pos[5]
                self.image_front.add_line(nn1, nn2)
            self.image_front.add_line(self.vertex_f_image_pos[i], self.vertex_f_image_pos[5])

        #self.image_front.append(draw.Text(self.id.__str__(), font_size=guide_font_size, x=self.vertex_b_image_pos[5][0], y=self.vertex_b_image_pos[5][1], text_anchor='middle'))

        #self.image_front.save_svg('test.svg')

    def drawCorrugation(self):
        if angle_min < self.in_angle:
            self.in_angle = angle_min

        A = np.arctan2(np.tan(self.angles[0] / 2), np.cos(self.in_angle))
        B = np.arctan2(np.tan(self.angles[1] / 2), np.cos(self.in_angle))
        C = np.arctan2(np.tan(self.angles[2] / 2), np.cos(self.in_angle))

        coeficents = np.array([self.b + self.c - self.a, self.c + self.a - self.b, self.a + self.b - self.c])
        max_index = (self.a, self.b, self.c).index(max(self.a, self.b, self.c))

        temp = coeficents[max_index]
        coeficents = temp / coeficents

        for i in reversed(range(1, small_iteration+1)):
            overflow = 2 * (A + B + C) - min(A, B, C) - np.deg2rad(270) - np.deg2rad(2)
            if overflow > 0:
                if max_index == 0:
                    A -= overflow / (coeficents.sum() * 2 - coeficents[max_index]) / i
                    B = np.arctan(np.tan(A) * (self.b + self.c - self.a) / (self.c + self.a - self.b))
                    C = np.arctan(np.tan(A) * (self.b + self.c - self.a) / (self.b + self.a - self.c))
                elif max_index == 1:
                    B -= overflow / (coeficents.sum() * 2 - coeficents[max_index]) / i
                    C = np.arctan(np.tan(B) * (self.a + self.c - self.b) / (self.b + self.a - self.c))
                    A = np.arctan(np.tan(B) * (self.a + self.c - self.b) / (self.b + self.c - self.a))
                elif max_index == 2:
                    C -= overflow / (coeficents.sum() * 2 - coeficents[max_index]) / i
                    A = np.arctan(np.tan(C) * (self.a + self.b - self.c) / (self.b + self.c - self.a))
                    B = np.arctan(np.tan(C) * (self.a + self.b - self.c) / (self.a + self.c - self.b))
            else:
                break

        if A*2 < self.angles[0]:
            print("How tf did this happen")
            print(self.angles_max)
            print(self.angles)
            print([A * 2, B * 2, C * 2])

        self.angles_max = [A * 2, B * 2, C * 2]

        self.drawImageBack()
        self.drawImageFront()

    def draw_bottom_flap(self, other):
        if self.pair.orientation == Orientation.RIGHT:
            bottom_point = other.vertex_b_image_pos[0] + np.array([np.linalg.norm(self.vertex_f_image_pos[1] - self.vertex_f_image_pos[2]), 0])
            section = [other.vertex_b_image_pos[0], bottom_point, other.vertex_b_image_pos[5], other.vertex_b_image_pos[6]]

            p = [other.vertex_b_image_pos[0], other.vertex_b_image_pos[6],
                 self.vertex_f_image_pos[1] - self.vertex_f_image_pos[5] + other.vertex_b_image_pos[0], bottom_point]
            t = ((p[0][0] - p[2][0]) * (p[2][1] - p[3][1]) - (p[0][1] - p[2][1]) * (p[2][0] - p[3][0])) / \
                ((p[0][0] - p[1][0]) * (p[2][1] - p[3][1]) - (p[0][1] - p[1][1]) * (p[2][0] - p[3][0]))
            if 0 <= t <= 1:
                section[2] = p[0] + (p[1] - p[0]) * t
                section = section[:3]
            else:
                p = [other.vertex_b_image_pos[5], other.vertex_b_image_pos[6],
                     self.vertex_f_image_pos[1] - self.vertex_f_image_pos[5] + other.vertex_b_image_pos[0], bottom_point]
                t = ((p[0][0] - p[2][0])*(p[2][1] - p[3][1]) - (p[0][1] - p[2][1])*(p[2][0] - p[3][0])) /\
                    ((p[0][0] - p[1][0])*(p[2][1] - p[3][1]) - (p[0][1] - p[1][1])*(p[2][0] - p[3][0]))
                if 0 <= t <= 1:
                    section[2] = p[0] + (p[1] - p[0]) * t

            mirror = np.array([[other.image_back.size[0] - v[0], v[1]] for v in section])
            self.image_front = copy_paste_polygon(self.image_front,
                                                          np.array([self.vertex_f_image_pos[1], self.vertex_f_image_pos[2]]),
                                                          other.image_back.mirror(),
                                                          mirror)

            #self.vertex_f_image_pos += offset
            self.image_front = copy_paste_polygon(self.image_front,
                                                          np.array(
                                                              [self.vertex_f_image_pos[3], self.vertex_f_image_pos[2]]),
                                                          other.image_back,
                                                          np.array(section))
            #self.vertex_f_image_pos += offset

            section = [self.vertex_f_image_pos[3], self.vertex_f_image_pos[4], self.vertex_f_image_pos[5], self.vertex_f_image_pos[2]]
            mirror = np.array([[self.image_front.size[0] - v[0], v[1]] for v in section])
            self.image_back = copy_paste_polygon(self.image_back,
                                                         np.array([self.vertex_b_image_pos[1], self.vertex_b_image_pos[2]]),
                                                         self.image_front.mirror(),
                                                         mirror)
            #self.vertex_b_image_pos += offset
        else:
            bottom_point = other.vertex_b_image_pos[1] + np.array([-np.linalg.norm(self.vertex_f_image_pos[0] - self.vertex_f_image_pos[4]), 0])
            section = [other.vertex_b_image_pos[1], bottom_point, other.vertex_b_image_pos[5], other.vertex_b_image_pos[6]]

            p = [other.vertex_b_image_pos[1], other.vertex_b_image_pos[6],
                 self.vertex_f_image_pos[0] - self.vertex_f_image_pos[5] + other.vertex_b_image_pos[1], bottom_point]
            t = ((p[0][0] - p[2][0]) * (p[2][1] - p[3][1]) - (p[0][1] - p[2][1]) * (p[2][0] - p[3][0])) / \
                ((p[0][0] - p[1][0]) * (p[2][1] - p[3][1]) - (p[0][1] - p[1][1]) * (p[2][0] - p[3][0]))
            if 0 <= t <= 1:
                section[2] = p[0] + (p[1] - p[0]) * t
                section = section[:3]
            else:
                p = [other.vertex_b_image_pos[5], other.vertex_b_image_pos[6],
                     self.vertex_f_image_pos[0] - self.vertex_f_image_pos[5] + other.vertex_b_image_pos[1], bottom_point]
                t = ((p[0][0] - p[2][0]) * (p[2][1] - p[3][1]) - (p[0][1] - p[2][1]) * (p[2][0] - p[3][0])) / \
                    ((p[0][0] - p[1][0]) * (p[2][1] - p[3][1]) - (p[0][1] - p[1][1]) * (p[2][0] - p[3][0]))
                if 0 <= t <= 1:
                    section[2] = p[0] + (p[1] - p[0]) * t

            self.image_front = copy_paste_polygon(self.image_front,
                                                          np.array([self.vertex_f_image_pos[0], self.vertex_f_image_pos[4]]),
                                                          other.image_back.mirror(),
                                                          np.array([[other.image_back.size[0] - v[0], v[1]] for v in section]))
            #self.vertex_f_image_pos += offset
            self.image_front = copy_paste_polygon(self.image_front,
                                                          np.array([self.vertex_f_image_pos[3], self.vertex_f_image_pos[4]]),
                                                          other.image_back,
                                                          np.array(section))
            #self.vertex_f_image_pos += offset

            section = [self.vertex_f_image_pos[3], self.vertex_f_image_pos[2], self.vertex_f_image_pos[5], self.vertex_f_image_pos[4]]
            self.image_back = copy_paste_polygon(self.image_back,
                                                         np.array([self.vertex_b_image_pos[0], self.vertex_b_image_pos[4]]),
                                                         self.image_front.mirror(),
                                                         np.array([[self.image_front.size[0] - v[0], v[1]] for v in section]))
            #self.vertex_b_image_pos += offset

    def draw_connecting_flap(self, other):
        if self.pair.orientation == Orientation.RIGHT and other.pair.orientation == Orientation.RIGHT:
            section = [other.vertex_b_image_pos[1], other.vertex_b_image_pos[2], other.vertex_b_image_pos[3], other.vertex_b_image_pos[5]]
            self.image_back = copy_paste_polygon(self.image_back,
                                                         np.array([self.vertex_b_image_pos[0], self.vertex_b_image_pos[4]]),
                                                         other.image_back,
                                                         np.array(section))
            #self.vertex_b_image_pos += offset
            center_point = rotate_vector((other.vertex_f_image_pos[5] - other.vertex_f_image_pos[4]) / 2, -other.angles_max[2] / 8 * 3) + other.vertex_f_image_pos[4]
            section = [other.vertex_f_image_pos[3], other.vertex_f_image_pos[4], center_point, other.vertex_f_image_pos[5], other.vertex_f_image_pos[2]]

            self.image_front = copy_paste_polygon(self.image_front,
                                                          np.array([self.vertex_f_image_pos[0], self.vertex_f_image_pos[4]]),
                                                          other.image_front,
                                                          np.array(section))
            #self.vertex_f_image_pos += offset

        elif self.pair.orientation == Orientation.LEFT and other.pair.orientation == Orientation.LEFT:
            section = [other.vertex_b_image_pos[0], other.vertex_b_image_pos[4], other.vertex_b_image_pos[3], other.vertex_b_image_pos[5]]
            self.image_back = copy_paste_polygon(self.image_back,
                                                         np.array([self.vertex_b_image_pos[1], self.vertex_b_image_pos[2]]),
                                                         other.image_back,
                                                         np.array(section))
            #self.vertex_b_image_pos += offset
            center_point = rotate_vector((other.vertex_f_image_pos[5] - other.vertex_f_image_pos[2]) / 2, other.angles_max[2] / 8 * 3) + other.vertex_f_image_pos[2]
            section = [other.vertex_f_image_pos[3], other.vertex_f_image_pos[2], center_point, other.vertex_f_image_pos[5], other.vertex_f_image_pos[4]]

            #section = [other.vertex_f_image_pos[3], other.vertex_f_image_pos[2], other.vertex_f_image_pos[5], other.vertex_f_image_pos[4]]
            self.image_front = copy_paste_polygon(self.image_front,
                                                          np.array([self.vertex_f_image_pos[1], self.vertex_f_image_pos[2]]),
                                                          other.image_front,
                                                          np.array(section))
            #self.vertex_f_image_pos += offset
        elif self.pair.orientation == Orientation.RIGHT and other.pair.orientation == Orientation.LEFT:
            section = [other.vertex_b_image_pos[4], other.vertex_b_image_pos[0], other.vertex_b_image_pos[5], other.vertex_b_image_pos[3]]
            self.image_back = copy_paste_polygon(self.image_back,
                                                         np.array([self.vertex_b_image_pos[0], self.vertex_b_image_pos[4]]),
                                                         other.image_back,
                                                         np.array(section))
            #self.vertex_b_image_pos += offset
            center_point = rotate_vector((other.vertex_f_image_pos[5] - other.vertex_f_image_pos[2]) / 2, other.angles_max[2] / 8 * 3) + other.vertex_f_image_pos[2]
            section = [other.vertex_f_image_pos[2], other.vertex_f_image_pos[3], other.vertex_f_image_pos[4], other.vertex_f_image_pos[5], center_point]

            #section = [other.vertex_f_image_pos[2], other.vertex_f_image_pos[3], other.vertex_f_image_pos[4], other.vertex_f_image_pos[5]]
            self.image_front = copy_paste_polygon(self.image_front,
                                                          np.array([self.vertex_f_image_pos[0], self.vertex_f_image_pos[4]]),
                                                          other.image_front,
                                                          np.array(section))
            #self.vertex_f_image_pos += offset
        elif self.pair.orientation == Orientation.LEFT and other.pair.orientation == Orientation.RIGHT:
            section = [other.vertex_b_image_pos[2], other.vertex_b_image_pos[1], other.vertex_b_image_pos[5], other.vertex_b_image_pos[3]]
            self.image_back = copy_paste_polygon(self.image_back,
                                                         np.array([self.vertex_b_image_pos[1], self.vertex_b_image_pos[2]]),
                                                         other.image_back,
                                                         np.array(section))
            #self.vertex_b_image_pos += offset
            center_point = rotate_vector((other.vertex_f_image_pos[5] - other.vertex_f_image_pos[4]) / 2, -other.angles_max[2] / 8 * 3) + other.vertex_f_image_pos[4]
            section = [other.vertex_f_image_pos[4], other.vertex_f_image_pos[3], other.vertex_f_image_pos[2], other.vertex_f_image_pos[5], center_point]

            #section = [other.vertex_f_image_pos[4], other.vertex_f_image_pos[3], other.vertex_f_image_pos[2], other.vertex_f_image_pos[5]]
            self.image_front = copy_paste_polygon(self.image_front,
                                                          np.array([self.vertex_f_image_pos[1], self.vertex_f_image_pos[2]]),
                                                          other.image_front,
                                                          np.array(section))
            #self.vertex_f_image_pos += offset



if __name__ == "__main__":
    t1 = Triangle(6.164012, 7.011076, 3.8463722, 1, 2, 3)
    t1.angles_max = [1.957931014042031, 1.9652505191637406, 1.97087972509901]
    t1.in_angle = np.deg2rad(50)

    t2 = Triangle(4.3164012, 3.3809501, 3.8463722, 2, 1, 4)
    t2.angles_max = [1.6072864127626334, 3.093210958990388, 1.8324241810503685]

    g = TrianglePair(t1, t2, [], [])
    g.set_orientation(Orientation.LEFT)
    g.draw_corougation()
    g.draw_image()
    g.image_front.refit_size()
    g.image_back.refit_size()
    g.image_back.draw_image(line_width).save_svg('test.svg')
    g.image_front.draw_image(line_width).save_svg('tes3.svg')
    #g.draw_corougation()
    #g.draw_image()




