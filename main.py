from vedo import *
import numpy as np
import TrianglePair as TP
from rectpack import newPacker
from PIL import Image, ImageDraw, ImageOps, ImageChops, ImageFont
from itertools import groupby
from SvgImage import *
import os

input_path = "heart.obj"
output_path = "heart_test.pdf"
output_dir = "cube"

height = 4
paper_size = (21, 29.7)
max_amount_of_paper = 50

tolerance = 0.1
angle_shorten = np.deg2rad(2)
angle_min = np.deg2rad(60)
angle_step = np.deg2rad(0.2)

push_tolerance = tolerance * 2
triangle_push_tolerance = tolerance * 2
same_point_tolerance = 0.7

small_iteration = 30

pixel_per_centimeter = 75
guide_font_size = 20
line_width = 2
padding = 20

def trimMesh2(vertices_, faces_):
    r = [[i] for i in range(len(vertices_))]
    v = [[v, False] for v in vertices_]
    for i in range(len(v)):
        if not v[i][1]:
            for j in range(i + 1, len(v)):
                if not v[j][1]:
                    d = np.linalg.norm(v[i][0] - v[j][0])
                    if d <= same_point_tolerance:
                        r[i].append(j)
                        v[j][1] = True
                    elif d <= push_tolerance:
                        v[j][0] = (v[j][0] - v[i][0]) / d * push_tolerance + v[i][0]
    '''
    for i in range(len(faces_)):
        AB = v[faces_[i][1]][0] - v[faces_[i][0]][0]
        BC = v[faces_[i][2]][0] - v[faces_[i][1]][0]
        CA = v[faces_[i][0]][0] - v[faces_[i][2]][0]
        c = np.linalg.norm(AB)
        a = np.linalg.norm(BC)
        b = np.linalg.norm(CA)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        dA = area * 2 / a - triangle_push_tolerance
        dB = area * 2 / b - triangle_push_tolerance
        dC = area * 2 / c - triangle_push_tolerance
        if dA < 0 and a >= b and a >= c:
            d = np.cross(np.cross(BC, -AB), BC)
            v[faces_[i][0]][0] += d / np.linalg.norm(d) * (-dA)
        if dB < 0 and b >= a and b >= c:
            d = np.cross(np.cross(CA, -BC), CA)
            v[faces_[i][0]][0] += d / np.linalg.norm(d) * (-dB)
        if dC < 0 and c >= a and c >= b:
            d = np.cross(np.cross(AB, -CA), AB)
            v[faces_[i][0]][0] += d / np.linalg.norm(d) * (-dC)'''

    vertices_ = np.array([v_[0] for v_ in v])

    for i in range(len(r)):
        for t in range(1, len(r[i])):
            vertices_ = np.delete(vertices_, r[i][t], axis=0)
            for x in range(0, len(faces_)):
                for y in range(0, len(faces_[x])):
                    if faces_[x][y] == r[i][t]:
                        faces_[x][y] = r[i][0]
                    elif faces_[x][y] > r[i][t]:
                        faces_[x][y] -= 1

            for y in range(t + 1, len(r[i])):
                if r[i][y] == r[i][t]:
                    r[i][y] = r[i][0]
                elif r[i][y] > r[i][t]:
                    r[i][y] -= 1
            for x in range(i + 1, len(r)):
                for y in range(0, len(r[x])):
                    if r[x][y] == r[i][t]:
                        r[x][y] = r[i][0]
                    elif r[x][y] > r[i][t]:
                        r[x][y] -= 1

    f = [len(list(set(x))) == 3 for x in faces_]
    faces_ = np.unique(faces_[f], axis=0)

    return vertices_, faces_


def pairTriangles(vertices_, faces_):

    nodes = [TP.Triangle(np.linalg.norm(vertices_[x[1]] - vertices_[x[2]]),
                         np.linalg.norm(vertices_[x[2]] - vertices_[x[0]]),
                         np.linalg.norm(vertices_[x[0]] - vertices_[x[1]]),
                         x[0],
                         x[1],
                         x[2]) for x in faces_]
    triangle_distance = []

    triangle_distance.append((nodes[0], 0))
    nodes.remove(nodes[0])

    counter = 0
    while counter < len(nodes):
        t = len(nodes)
        while counter < t:
            triangle = triangle_distance[counter][0]
            triangles = filter(lambda x: x.adjacent(triangle), nodes)
            triangles = sorted(list(triangles), key=lambda x: np.linalg.norm(vertices_[triangle.adjacent_edge(x)[0]] - vertices_[triangle.adjacent_edge(x)[1]]))
            for i in triangles:
                nodes.remove(i)
                triangle_distance.append((i, triangle_distance[counter][1] + 1))
            counter += 1

    #print(nodes)

    unpaired = []
    pairs = []

    for i in range(0, len(triangle_distance)):
        if len(triangle_distance) > 0:
            t1 = triangle_distance[0]
            triangle_distance.remove(t1)
            #print(t1)
            e = list(filter(lambda x: x[0].adjacent(t1[0]), triangle_distance))
            if len(e) > 0:
                t2 = e[0]
                triangle_distance.remove(t2)
                #print(t2)
                pairs.append([t1[0], t2[0]])
            else:
                unpaired.append(t1[0])
                #print("unpaired", t1)

        else:
            break

    #print("pairs", len(pairs))
    #print("unpairs", len(unpaired))

    def pos(tri):
        return (vertices_[tri.A] + vertices_[tri.B] + vertices_[tri.C]) / 3

    for i in range(len(unpaired)):
        if len(unpaired) >= 2:
            start = unpaired[0]
            end = unpaired[1]
            # A*
            openpath = [[[None, start], 0, 0, []]]
            explored_paths = []
            for j in range(len(pairs) * 2):
                if not openpath[0][0][1].adjacent(end):
                    s = openpath[0]
                    openpath.remove(s)
                    avalible_paths = list(filter(lambda x: x not in s[3] and [x[1], x[0]] not in s[3], pairs))
                    a1 = list(filter(lambda x: x[0].adjacent(s[0][1]) and x not in explored_paths, avalible_paths))
                    a2 = list(filter(lambda x: x[1].adjacent(s[0][1]) and [x[1], x[0]] not in explored_paths, avalible_paths))
                    a2 = [[a[1], a[0]] for a in a2]
                    a1.extend(a2)

                    explored_paths.extend(a1)
                    p = []
                    for a in a1:
                        path_to_node = s[3].copy()
                        path_to_node.append(a)
                        p.append([a, np.linalg.norm(pos(a[1]) - pos(end)) + s[2] + 0.1, s[2] + 0.1, path_to_node])

                    for i in p:
                        index = 0
                        for t in range(len(openpath)):
                            if openpath[t][1] > i[1]:
                                index = t + 1
                                break
                        openpath.insert(index, i)
                else:
                    #print(len(pairs))
                    path = openpath[0][3]
                    #print(path)
                    for i in path:
                        if i in pairs:
                            pairs.remove(i)
                        elif [i[1], i[0]] in pairs:
                            pairs.remove([i[1], i[0]])

                    #print(len(path))
                    new_pairs = [[start, path[0][0]]]
                    for i in range(len(path)-1):
                        new_pairs.append([path[i][1], path[i+1][0]])
                    new_pairs.append([path[len(path) - 1][1], end])
                    pairs.extend(new_pairs)
                    unpaired.remove(start)
                    unpaired.remove(end)
                    #print(len(pairs))
                    break
        else:
            break


    # align pairs
    id_counter = 0
    for p in pairs:
        p[0].id = id_counter
        p[1].id = id_counter + 1
        id_counter += 2
        x1 = p[0].edges()
        x2 = p[1].edges()
        x = [t for t in x1 if t in x2]
        x = x[0]
        for i in range(3):
            if x != p[0].edges()[0]:
                p[0].rotateClockwise()
            if x != p[1].edges()[0]:
                p[1].rotateClockwise()

    # set orientations
    triangle_pairs = []
    for p in pairs:
        tri1 = []
        tri2 = []
        for i in pairs:
            if i[0].adjacent(p[0]) and i[0] != p[1] and i[0] not in tri1:
                tri1.append(i[0])
            if i[1].adjacent(p[0]) and i[1] != p[1] and i[1] not in tri1:
                tri1.append(i[1])

            if i[0].adjacent(p[1]) and i[0] != p[0] and i[0] not in tri2:
                tri2.append(i[0])
            if i[1].adjacent(p[1]) and i[1] != p[0] and i[1] not in tri2:
                tri2.append(i[1])

        if len(tri1) != 2 or len(tri2) != 2:
            print("pair not connected to exctaly 4 triangles")
            print("t1", p[0], [t.__str__() for t in tri1])
            print("t2", p[1], [t.__str__() for t in tri2])
        triangle_pairs.append(TP.TrianglePair(p[0], p[1], tri1, tri2))
    triangle_pairs[0].set_orientation(TP.Orientation.LEFT)

    print("pairs", len(pairs))
    print("unpairs", len(unpaired))

    return triangle_pairs


def calculateAngleBound2(vertices_, trianglePairs_):

    def normVector(triangle):
        AB = vertices_[triangle.B] - vertices_[triangle.A]
        AC = vertices_[triangle.C] - vertices_[triangle.A]
        return np.cross(AB, AC) / np.linalg.norm(np.cross(AB, AC))

    for p in trianglePairs_:
        for t in p.connectedtriangles1:
            if t.adjacent(p.triangle1):
                t1N = normVector(p.triangle1)
                t2N = normVector(t)
                a = np.pi - np.arccos(np.dot(t1N, t2N))
                if a / 2 - angle_shorten/2 < p.triangle1.in_angle:
                    p.triangle1.in_angle = a/2 - angle_shorten/2
        for t in p.connectedtriangles2:
            if t.adjacent(p.triangle2):
                t1N = normVector(p.triangle2)
                t2N = normVector(t)
                a = np.pi - np.arccos(np.dot(t1N, t2N))
                if a / 2 - angle_shorten/2 < p.triangle2.in_angle:
                    p.triangle2.in_angle = a/2 - angle_shorten/2


def calculateAngleBound(vertices_, trianglePairs_):
    for v in range(len(vertices_)):
        triangles = []
        for p in trianglePairs_:
            if v in p.triangle1.vertices():
                triangles.append([p.triangle1, p.triangle1.vertices().index(v)])
            if v in p.triangle2.vertices():
                triangles.append([p.triangle2, p.triangle2.vertices().index(v)])

        angles = np.array([t[0].angles[t[1]] for t in triangles])
        angle_sum = angles.sum()
        x = 0
        prev_s = 100000
        s = 0
        short = 0
        while x <= 10 * np.pi:
            s = 0
            for a in angles:
                if 1 - np.cos(x + a) == 0 or abs(1 - 2 * (1 - np.cos(a)) / (1 - np.cos(x + a))) > 1:
                    break
                s += np.arccos(1 - 2 * (1 - np.cos(a)) / (1 - np.cos(x + a)))
            if s <= 2 * np.pi or prev_s < s:
                #short = max(angle_tolerance - (s - 2 * np.pi) / len(angles), 0)
                break
            else:
                prev_s = s
                x += angle_step
        for e, t in enumerate(triangles):
            t[0].angles_max[t[1]] = min(t[0].angles[t[1]] + x, np.pi)


def drawCorougation(trianglePairs_):
    for p in trianglePairs_:
        p.draw_corougation()
    for p in trianglePairs_:
        p.draw_connections()
    for p in trianglePairs_:
        p.draw_image()


def packIntoPDF(trianglePairs_):
    size = (int(paper_size[0] * pixel_per_centimeter), int(paper_size[1] * pixel_per_centimeter))
    packer = newPacker()
    for i in range(len(trianglePairs_)):
        trianglePairs_[i].image_front.refit_size()
        trianglePairs_[i].image_back.refit_size()
        packer.add_rect(trianglePairs_[i].image_front.size[0] + padding * 2, trianglePairs_[i].image_front.size[1] + padding * 2, rid=i * 2)
        packer.add_rect(trianglePairs_[i].image_back.size[0] + padding * 2, trianglePairs_[i].image_back.size[1] + padding * 2, rid=i * 2 + 1)
    packer.add_bin(size[0], size[1], count=max_amount_of_paper)

    packer.pack()
    pages = []
    rects = packer.rect_list()

    for i in range(len(packer)):
        image = SvgImage(np.array([0, 0]), [])
        pages.append(image)

    trianglePairs_[0].image_front.draw_image(line_width).save_svg(output_dir + "/" + "0F.svg")
    trianglePairs_[0].image_back.draw_image(line_width).save_svg(output_dir + "/" + "0B.svg")

    for rect in rects:
        b, x, y, w, h, rid = rect
        if rid % 2 == 0:
            if w - padding * 2 != trianglePairs_[int(rid / 2)].image_front.size[0] or\
                    h - padding * 2 != trianglePairs_[int(rid / 2)].image_front.size[1]:

                trianglePairs_[int(rid / 2)].image_front.rotate(np.deg2rad(90), trianglePairs_[int(rid / 2)].image_front.size / 2)
                trianglePairs_[int(rid / 2)].image_front.refit_size()
                trianglePairs_[int(rid / 2)].image_front.translate(np.array([x + padding, y + padding]))
                pages[b].impose(trianglePairs_[int(rid / 2)].image_front)
            else:
                trianglePairs_[int(rid / 2)].image_front.translate(np.array([x + padding, y + padding]))
                pages[b].impose(trianglePairs_[int(rid / 2)].image_front)
        else:
            if w - padding * 2 != trianglePairs_[int(rid / 2)].image_back.size[0] or\
                    h - padding * 2 != trianglePairs_[int(rid / 2)].image_back.size[1]:

                trianglePairs_[int(rid / 2)].image_back.rotate(np.deg2rad(90), trianglePairs_[int(rid / 2)].image_back.size / 2)
                trianglePairs_[int(rid / 2)].image_back.refit_size()
                trianglePairs_[int(rid / 2)].image_back.translate(np.array([x + padding, y + padding]))
                pages[b].impose(trianglePairs_[int(rid / 2)].image_back)
            else:
                trianglePairs_[int(rid / 2)].image_back.translate(np.array([x + padding, y + padding]))
                pages[b].impose(trianglePairs_[int(rid / 2)].image_back)

    isExist = os.path.exists(output_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(output_dir)

    for i in range(len(pages)):
        pages[i].refit_size()
        pages[i].draw_image(line_width).save_svg(output_dir + "/" + i.__str__() + ".svg")
    pages[0].save(output_path, resolution=pixel_per_centimeter * 2.54, save_all=True, append_images=pages[1:])


if __name__ == '__main__':
    mesh = Mesh(input_path)
    mesh.triangulate()

    normals = np.array(mesh.vertex_normals)
    faces = np.array(mesh.faces())
    vertices = np.array(mesh.vertices)

    for i in range(len(faces)):
        if np.dot(
                np.cross(vertices[faces[i][1]] - vertices[faces[i][0]], vertices[faces[i][2]] - vertices[faces[i][1]]),
                normals[faces[i][1]]) < 0:
            temp = faces[i][0]
            faces[i][0] = faces[i][2]
            faces[i][2] = temp

    vertices -= vertices.min(axis=0)
    vertices /= vertices.max(axis=0)[1] / height

    print(len(vertices))
    print(len(faces))

    vertices, faces = trimMesh2(vertices, faces)
    print(len(vertices))
    print(len(faces))

    pairs = pairTriangles(vertices, faces)

    #calculateAngleBound(vertices, pairs)
    calculateAngleBound2(vertices, pairs)

    drawCorougation(pairs)

    packIntoPDF(pairs)

    f = []
    for p in pairs:
        f.append(p.triangle1.vertices())
        f.append(p.triangle2.vertices())

    mesh_new = Mesh([vertices, f])
    rgba = np.ones((mesh_new.ncells, 3)) * 255  # RGBA values
    for i in pairs:
        a = -1
        b = -1
        for j, x in enumerate(f):
            if i.triangle1.A in x and i.triangle1.B in x and i.triangle1.C in x and a == -1:
                a = j
            if i.triangle2.A in x and i.triangle2.B in x and i.triangle2.C in x and b == -1:
                b = j
            if a != -1 and b != -1:
                break
        if i.orientation == TP.Orientation.RIGHT:
            rgba[a] = np.array([255, 0, 0])
            rgba[b] = np.array([255, 0, 0])
        elif i.orientation == TP.Orientation.LEFT:
            rgba[a] = np.array([0, 0, 255])
            rgba[b] = np.array([0, 0, 255])
        else:
            rgba[a] = np.array([0, 255, 0])
            rgba[b] = np.array([0, 255, 0])
    mesh_new.cellcolors = rgba
    mesh_new.linecolor([0, 0, 0])
    mesh_new.linewidth(0.5)
    mesh_new = mesh_new.compute_normals()
    cell_ids = mesh_new.labels('id', on="cells").c('black')

    show(mesh_new, cell_ids, axes=1)
