import numpy as np 
from matplotlib import cm
import cv2
from scipy.interpolate import splprep, splev
import arrow
from termcolor import cprint 

import matplotlib.pyplot as plt
import itertools, shapely
from shapely.geometry import LineString

class Camera():
    camera_matrix = np.array([525, 0.0, 340, 0.0, 525, 230, 0.0, 0.0, 1.0]).reshape(3,3)
    camera_height = 480
    camera_width = 640


COLORS = cm.get_cmap("Set1", 10)


def projection(camera_pose, points_3d):
    T = np.linalg.inv(camera_pose)
    tvec =np.array(T[0:3, 3])
    rvec, _ = cv2.Rodrigues(T[:3,:3])

    point2d = cv2.projectPoints(np.array(points_3d), rvec, tvec, Camera.camera_matrix, None)
    point2d = point2d[0].squeeze()
    points_ref = []
    for p in point2d:
        i, j = [round(p[1]), round(p[0])]
        if i < Camera.camera_height and i >= 0 and j < Camera.camera_width and j >= 0:
            points_ref.append([j,i])

    return np.array(points_ref)



def getColors(colormap_name = "Set1", number = 10):
    colors = cm.get_cmap(colormap_name, number) 
    return [(colors(i)[0] * 255, colors(i)[1] * 255, colors(i)[2] * 255)  for i in range(20)]


def samplePaths(paths, density=10, drop_endpoints=False, n_min_path=5):
    #vertices, edges = [], []
    excluded = []
    vertices_dict = {}
    edges_dict = {}
    for id, p in paths.items():
        indeces = np.linspace(0, len(p)-1, int(len(p)/density)+1)
        indeces = indeces.astype(np.int)
        p_new = [pp for i, pp in enumerate(p) if i in indeces]
        
        if drop_endpoints:
            p_new = p_new[1:-1]

        if len(p_new) >= n_min_path:
            #vertices.append(p_new)
            vertices_dict[id] = p_new

            e_new = [[i-1, i+1] for i in range(1, len(p_new)-1)]
            e_new.insert(0, [1])
            e_new.append([len(p_new)-2])
            #edges.append(e_new)
            edges_dict[id] = e_new
        else:
            #print("path {} too short, added to the list of excluded paths.".format(id))
            excluded.append(id)

    #print("excluded paths: ", excluded)
    return vertices_dict, edges_dict, excluded


def labelVertices(vertices_list, instances):
    labels = {}
    for c, vertices in vertices_list.items():
        labels_tmp = []
        for iter in range(len(vertices)):
            for id, mask in enumerate(instances):
                if mask[tuple(vertices[iter])] == 255:
                    labels_tmp.append(id)
                    break
        labels[c] = int(np.median(labels_tmp))
    return labels


def computeRadii(paths_dict, dist_img): 
    return {k: estimateRadiusFromSegment(path, dist_img) for k, path in paths_dict.items()}


def estimateRadiusFromSegment(path, dist_img, min_px = 3):
    values = [dist_img[tuple(p)] for p in path]
    rv = np.mean(values) if values else min_px
    return rv


def computeSplines(paths_dict, num_points=1000, s=0.0, key=None):
    splines = {}
    for it, path in paths_dict.items():  
        radius = -1
        if key != None:
            radius = path["radius"]
            path = path[key]
        

        if len(path) > 3:
            points, der, der2 = computeSpline(path, num_points=num_points, s=s)
            splines[it] = {"points": points, "der": der, "der2": der2 , "radius": radius}
        else:
            der = [[0.01,0.01] for p in path]
            der2 = [[0.01,0.01] for p in path]
            splines[it] = {"points": path, "der": der, "der2": der2, "radius": radius}

    '''
    for k, v in splines.items():
        xs, ys = zip(*v["points"])
        plt.plot(xs, ys, label=str(k))

        xp, yp = zip(*paths_dict[k])
        plt.scatter(xp, yp, label=str(k))

    plt.show()
    '''

    return splines


def computeSpline(points, num_points = 10, k = 3, s = 0.0):     
    points = np.array(points)
    tck, u = splprep(points.T, u=None, k=k, s=s, per=0)
    u_new = np.linspace(u.min(), u.max(), num_points)
    
    x_, y_ = splev(u_new, tck, der=0)

    xd_, yd_ = splev(u_new, tck, der=1)

    xdd_, ydd_ = splev(u_new, tck, der=2)

    return np.column_stack((x_,y_)), np.column_stack((xd_,yd_)), np.column_stack((xdd_,ydd_))


def roundRadius(radius, mul=1.0, bound=6):
    r = int(np.ceil(radius*mul))
    if r < bound: 
        r = bound
    return r


def colorMasks(splines, shape, mask_input=None):
    colors_k = [i for i in range(10)]
    colors_k_excl = [i for i in range(10) if i not in list(splines.keys())]
    colors = COLORS
    if mask_input is None:
        mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:
        mask = cv2.cvtColor(mask_input, cv2.COLOR_GRAY2BGR)
    
    for k, v in splines.items():
        #t0 = arrow.utcnow()
        if k in colors_k:
            c = (int(colors(k)[0]*255), int(colors(k)[1]*255), int(colors(k)[2]*255))
            colors_k.remove(k)
        else:
            if colors_k_excl:
                rand_k = np.random.choice(colors_k_excl)
                c = (int(colors(rand_k)[0]*255), int(colors(rand_k)[1]*255), int(colors(rand_k)[2]*255))
                colors_k.remove(rand_k)
                colors_k_excl.remove(rand_k)
            else:
                c = (127,127,127)


        r = roundRadius(v["radius"])  

        for p in splines[k]["points"]:
            cv2.circle(mask, (int(p[1]), int(p[0])), r, c, -1)

        if False:
            new_points = splineExtension(splines[k]["points"], shape=mask.shape[:2])
            for p in new_points:
                p = (int(p[1]), int(p[0]))
                if np.sum(mask[p[1], p[0]]) == 0:
                    cv2.circle(mask, p, r, c, -1)

        #print("coloring {0}, time: {1:.4f} ms".format(k, (arrow.utcnow() - t0).total_seconds() * 1000))

    mask[mask_input == 0] = (0,0,0)
    return mask




def checkBoundsInv(value, bound_low, bound_up):
    if value < bound_up and value > bound_low:
        return False
    return True


def splineExtension(spline_points, shape, step=3, margin=100):

    new_points = []
    #print(spline_points[0], spline_points[-1])
    if checkBoundsInv(spline_points[0][0], margin, shape[0]-margin) or checkBoundsInv(spline_points[0][1], margin, shape[1]-margin):
        point_0 = np.array(spline_points[0])
        dir_0 = point_0 - np.array(spline_points[1])
        dir_0 = dir_0 / np.linalg.norm(dir_0)

        for i in range(10):
            x = point_0[0] + i*step * dir_0[0]
            y = point_0[1] + i*step * dir_0[1]
            if x > 0 and x < shape[0] and y > 0 and y < shape[1]:
                new_points.append([x,y])


    if checkBoundsInv(spline_points[-1][0], margin, shape[0]-margin) or checkBoundsInv(spline_points[-1][1], margin, shape[1]-margin):
        point_1 = np.array(spline_points[-1])
        dir_1 = point_1 - np.array(spline_points[-2])
        dir_1 = dir_1 / np.linalg.norm(dir_1)

        for i in range(10):
            x = point_1[0] + i*step * dir_1[0]
            y = point_1[1] + i*step * dir_1[1]
            if x > 0 and x < shape[0] and y > 0 and y < shape[1]:
                new_points.append([x,y])

    return new_points


def distance2D(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5


def intersectionSplines(splines, paths, single_degree_nodes, img):

    keys = list(splines.keys())
    combs = list(map(list, itertools.combinations(keys, 2)))

    #t0 = arrow.utcnow()

    out = []
    for it0, it1 in combs:
        point0 = splines[it0]["points"]
        radius0 = splines[it0]["radius"]
        point1 = splines[it1]["points"]
        radius1 = splines[it1]["radius"]

        line0 = LineString(point0)
        line1 = LineString(point1)
        intersect_point = line0.intersection(line1)

        if type(intersect_point) == shapely.geometry.multipoint.MultiPoint:
            intersect_points = [(point.x, point.y) for point in intersect_point]
        elif type(intersect_point) == shapely.geometry.point.Point:
            intersect_points = [(intersect_point.x, intersect_point.y)]
        else:
            intersect_points = None

        if intersect_points is None:
            continue
        
        for intersect_point in intersect_points:
            path0 = paths[it0]["nodes"]
            path0p = paths[it0]["points"]
            path1 = paths[it1]["nodes"]
            path1p = paths[it1]["points"]

            candidate0, candidate1 = [], []
            for n in single_degree_nodes:
                if n in path0:
                    pos = path0p[path0.index(n)]
                    dist = distance2D(pos, intersect_point)
                    candidate0.append({"n": n, "dist": dist})

                elif n in path1:
                    pos = path1p[path1.index(n)]
                    dist = distance2D(pos, intersect_point)
                    candidate1.append({"n": n, "dist": dist})



            candidate0_sorted = sorted(candidate0, key=lambda d: d['dist'], reverse=False) 
            candidate1_sorted = sorted(candidate1, key=lambda d: d['dist'], reverse=False) 

            c0 = [c["n"] for c in candidate0_sorted[:2]]
            c1 = [c["n"] for c in candidate1_sorted[:2]]

            c = [int(intersect_point[0]), int(intersect_point[1])]
            img_crop = img[c[0]-2:c[0]+3, c[1]-2:c[1]+3].reshape(-1, 3).astype(np.float32)
            pos_int_color = img_crop.mean(axis=0)

            #print(c, pos_int_color)
            if False:
                canvas = img.copy()
                cv2.circle(canvas, tuple([coord[1], coord[0]]), 3, (255, 255, 59), -1)
                cv2.imshow("canvas", canvas)
                cv2.waitKey(0)
                print(pos_int_color)
                print(coord)

            out.append({"c0": c0, "c1": c1, "it0": it0, "it1": it1, "radius0": radius0, "radius1": radius1, 
                        "pos_int": np.array(intersect_point) / img.shape[:2], "color_int": pos_int_color / 255,
                        "score0": 0.0, "score1": 0.0})

    #print("time: ", (arrow.utcnow() - t0).total_seconds()*1000) 
    return out



def stdColorsBetweenPoints(point0, point1, image):

    pos0, pos00 = np.array(point0), np.array(point1)
    dir = pos00 - pos0
    N = np.max(np.abs(dir))
    N_arr = np.arange(0,N,1).reshape(-1,1)
    steps = (np.ones((N,2), dtype=np.float32) * np.hstack([N_arr, N_arr])) * dir/N
    locations = pos0 + steps
    locations = locations.astype(np.int32)        
    values = image[locations[:,0], locations[:,1]]
    return np.std(values, axis=0), locations 


def colorIntersection(mask, points, radius, key):
    color = COLORS(key)
    c = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
    r = roundRadius(radius)
    x, y = zip(*points)
    for i in range(len(x)):
        cv2.circle(mask, (int(y[i]), int(x[i])), r, c, -1)


def intersectionScoresFromColor(data_list, nodes, image, colored_mask=None, debug=False):

    for v in data_list:
        # 0
        attr0, attr00 = nodes[v["c0"][0]], nodes[v["c0"][1]]
        std0, locs0 = stdColorsBetweenPoints(attr0["pos"], attr00["pos"], image)
 
        # 1
        attr1, attr11 = nodes[v["c1"][0]], nodes[v["c1"][1]]
        std1, locs1 = stdColorsBetweenPoints(attr1["pos"], attr11["pos"], image)

        if debug:
            print("xxxxxxxxxxxxxxxxxxxxxxx")
            print(std0, np.mean(std0))
            print(std1, np.mean(std1))
            print("xxxxxxxxxxxxxxxxxxxxxxx")

        if np.mean(std0) < np.mean(std1):
            score0 = 1000
            score1 = 0
        else:
            score0 = 0
            score1 = 1000        

        v["score0"] = score0
        v["score1"] = score1

        if colored_mask is not None:  
            if v["score0"] > v["score1"]:
                colorIntersection(colored_mask, locs1, radius=v["radius1"], key=v["it1"])
                colorIntersection(colored_mask, locs0, radius=v["radius0"], key=v["it0"])
            else:
                colorIntersection(colored_mask, locs0, radius=v["radius0"], key=v["it0"])
                colorIntersection(colored_mask, locs1, radius=v["radius1"], key=v["it1"])

    return data_list
