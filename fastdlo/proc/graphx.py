import numpy as np
import math, copy, shapely, itertools
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import networkx as nx

import arrow
from termcolor import cprint

class Graph(object):

    def __init__(self, vertices, edges, image, dist_img, radius=None, labels=None, compute_rgb_mean=True):
        
        self.shape = image.shape[:2]
        self.image = image
        self.dist_img = dist_img
        self.num_segments = len(list(vertices.keys()))

        self.vertices_dict_nodes = {}
        self.nodes = {}
        self.single_nodes = []

        offset = 0
        for cc, vertices_raw in vertices.items():

            if radius is None: rad = 5 
            else: rad = radius[cc]

            if labels is None: lab = 0 
            else: lab = labels[cc]

            rad_int = int(round(rad))
            rad = rad / self.shape[0]

            vertices_nodes = []


            # endpoints
            for iter in range(len(vertices_raw)):
                t, t_nn = None, None
                if iter == 0:
                    t, t_nn = 0, 1 
                elif iter == len(vertices_raw)-1:
                    t, t_nn = -1, -2
                
                if t is not None:
                    pos = vertices_raw[t]
                    pos_norm = pos / self.shape
                    pos_nn = vertices_raw[t_nn]
                    pos_norm_nn = pos_nn / self.shape

                    dir = np.array(pos_norm) - np.array(pos_norm_nn)
                    dir = dir / np.linalg.norm(dir)

                    if compute_rgb_mean:
                        img_crop = self.image[pos[0]-rad_int:pos[0]+rad_int, pos[1]-rad_int:pos[1]+rad_int].reshape(-1, 3).astype(np.float32)
                        if img_crop.shape[0] > 0:
                            color = img_crop.mean(axis=0) / 255
                        else:
                            color = self.image[tuple(pos)] / 255
                    else:
                        color = self.image[tuple(pos)] / 255

                    self.nodes[iter + offset] = {"pos": pos, "pos_norm": pos_norm, "pos_d": dir, "radius": rad, "label": lab, "segment": cc, "color": color}            
                    self.single_nodes.append(iter + offset)
                else:
                    self.nodes[iter + offset] = {"pos": vertices_raw[iter]}
                vertices_nodes.append(iter + offset)

            self.vertices_dict_nodes[cc] = {"nodes": vertices_nodes, "pos": vertices_raw}
            offset += len(vertices_raw)    
   
    def distance2D(self, point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    def idMinimumDistancePointToList(self, point, list_points):
        diff = (np.array(list_points) - np.array(point))**2
        min_id = np.where(diff <= np.min(diff) + 0.00001)[0][0]
        return min_id

    def idMinimumDistancePointToListSlow(self, point, list_points):
        distances = [self.distance2D(point, p) for p in list_points]
        #print(np.min(distances))
        return np.argmin(distances)

    def findNearest(self, array, value):     
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()     
        return idx


    def computeSupervisionEdgesFromGT(self, gt_data, add_intersection_node = False, debug = False):

        labels_gt = [0, 1]
        line1 = LineString([(p[0], p[1]) for p in gt_data[labels_gt[0]]])
        line2 = LineString([(p[0], p[1]) for p in gt_data[labels_gt[1]]])
        intersect_points = line1.intersection(line2)

        if type(intersect_points) == shapely.geometry.multipoint.MultiPoint:
            points_out = [(point.x, point.y) for point in intersect_points]
        elif type(intersect_points) == shapely.geometry.point.Point:
            points_out = [(intersect_points.x, intersect_points.y)]
        else:
            points_out = []

        # single degree nodes
        nodes_1 = [node for (node, val) in self.g.degree() if val == 1]
        if debug: print("nodes_1: ", nodes_1)
        
        # dict of sequence of endpoints from gt data
        nodes_dict = {l: {} for l in labels_gt}
        for v in nodes_1:
            l = self.g.nodes[v]["label"]
            p = self.g.nodes[v]["pos"]
            query = [p[1], p[0]]
            id = self.idMinimumDistancePointToListSlow(query, gt_data[l])
            nodes_dict[l][v] = id

        for k, v in nodes_dict.items():
            dict_sorted = sorted(v.items(), key=lambda x: x[1], reverse=False)[1:-1]
            nodes_dict[k] = {x[0]: x[1] for x in dict_sorted}
        
        if debug: print("nodes_dict: ", nodes_dict)

        pos_edges_overall, neg_edges_overall = [], []
        new_id = len(self.g.nodes())
        for point in points_out:
            if debug: print("POINT INT: ", point)  

            ids_point_to_gt = {l: {} for l in labels_gt}
            for k, v in ids_point_to_gt.items():
                    ids_point_to_gt[k] = self.idMinimumDistancePointToListSlow(point, gt_data[k])
            if debug: print("ids_point_to_gt: ", ids_point_to_gt)

            try:
                nodes_int = {l: [] for l in labels_gt}
                for k, v in nodes_dict.items():
                    list_ids = list(v.values())
                    if len(list_ids) > 0:
                        idx = self.findNearest(list_ids, ids_point_to_gt[k])
                        if (list_ids[idx] - ids_point_to_gt[k]) < 0 or idx == 0:
                            idxs = [idx, idx+1]
                        else:
                            idxs = [idx-1, idx]
                        list_nodes = list(v.keys())
                        nodes_int[k] = [list_nodes[idxs[0]], list_nodes[idxs[1]]]
            except:
                cprint("SKIIIPPPP" ,"red")

            if debug: print("nodes_int: ", nodes_int)

                
            pos_edges, neg_edges = self.computePosNegEdges(nodes_int)
            # update overall lists
            pos_edges_overall.extend(pos_edges)
            neg_edges_overall.extend(neg_edges)

            #print(pos_edges, neg_edges)

            if add_intersection_node:
                node_edges = []
                for e1,e2 in pos_edges:
                    node_edges.append(e1)
                    node_edges.append(e2)
                node_edges = list(set(node_edges)) 

                self.addIntersectionNode(point, node_edges, node_id=new_id)
                new_id += 1

        # degenerate case where len(points_out) == 0:
        if len(pos_edges_overall) == 0:
            nodes_int = {l: [] for l in labels_gt}
            for k, v in nodes_dict.items():
                nodes_int[k] = list(v.keys())
            
            if debug: print("nodes_int deg: ", nodes_int)

            pos_edges, neg_edges = self.computePosNegEdgesDegenerative(nodes_int)
            # update overall lists
            pos_edges_overall.extend(pos_edges)
            neg_edges_overall.extend(neg_edges)

        if add_intersection_node:
            self.g.add_nodes_from(self.new_nodes_list)
            self.g.add_edges_from(self.new_edges_list, label=0)      


        return pos_edges_overall, neg_edges_overall

    def computePosNegEdges(self, nodes_dict, debug=False):
        # check existance of path, otherwise add positive edge
        pos_edges = []
        for k, nodes in nodes_dict.items():
            if len(nodes) == 2:
                node_0 = nodes[0]
                node_1 = nodes[1]
                if not nx.has_path(self.g, node_0, node_1):
                    pos_edges.append((node_0, node_1))
                    pos_edges.append((node_1, node_0))

        if debug: print("pos: ", pos_edges)

        # negative edges
        nodes = set([p for pair in pos_edges for p in pair])
        combinations_list = list(map(list, itertools.combinations(nodes, 2)))
        neg_edges = [tuple(c) for c in combinations_list if tuple(c) not in pos_edges]
        if debug: print("neg: ", neg_edges)
        
        return pos_edges, neg_edges
    
    def computePosNegEdgesDegenerative(self, nodes_dict, debug=False):
        # check existance of path, otherwise add positive edge
        pos_edges = []
        for k, nodes in nodes_dict.items():
            for it in range(1, len(nodes)):
                node_0 = nodes[it-1]
                node_1 = nodes[it]
                if not nx.has_path(self.g, node_0, node_1):
                    #print("added: ", node_0, node_1)
                    pos_edges.append((node_0, node_1))
                    pos_edges.append((node_1, node_0))

        if debug: print("pos: ", pos_edges)

        # negative edges
        nodes = set([p for pair in pos_edges for p in pair])
        combinations_list = list(map(list, itertools.combinations(nodes, 2)))
        neg_edges_init = {}
        for c in combinations_list:
            if tuple(c) not in pos_edges:
                if self.g.nodes[c[0]]["label"] != self.g.nodes[c[1]]["label"]:
                    dist = self.distance2D(self.g.nodes[c[0]]["pos"], self.g.nodes[c[1]]["pos"])
                    neg_edges_init[(c[0], c[1])] = dist


        if len(list(neg_edges_init.keys())) > len(pos_edges):
            dict_sorted = sorted(neg_edges_init.items(), key=lambda x: x[1], reverse=False)[:len(pos_edges)]
            neg_edges = [v[0] for v in dict_sorted]
        else:
            neg_edges = list(neg_edges_init.keys())
        
        
        if debug: print("neg: ", neg_edges)
        
        
        return pos_edges, neg_edges


    def computeLocalPredictionEdges(self, int_points_dict, excluded_paths, debug=False):
        
        if debug:
            print("INT POINTS: ", int_points_dict)
            print("excluded: ", excluded_paths)

        
        # step 1: modify int points dict to manage excluded paths

        int_dict = copy.deepcopy(int_points_dict)
        for p in excluded_paths:
            if debug: print("P: ", p)
            intersections_affected = [k for k,v in int_dict.items() if p in v["segments"]]

            if debug: print("intersectiosn affected: ", intersections_affected)
            if len(intersections_affected) > 0:
                # merge affected intersections
                int_points = [int_dict[k]["point"] for k in intersections_affected]
                mean_int = np.mean(int_points, axis=0)
                segments_ids = list(set([s for k in intersections_affected for s in int_dict[k]["segments"] if s != p]))
                int_id = intersections_affected[0]            

                int_dict[int_id] = {"point": mean_int, "segments": segments_ids}

                for i in range(1, len(intersections_affected)):
                    del int_dict[intersections_affected[i]]
        
        #for i in ints_to_delete:
        #    del int_dict[i]
            
        if debug: print("NEW INT POINTS: ", int_dict)

        # step 3: loop for each intersection
        overall_edges = []
        for k, v in int_dict.items():
            int_point = v["point"]
            seg_ids = v["segments"]

            nodes_dict = {}
            for n in self.single_nodes:
                s = self.nodes[n]["segment"]
                if s in seg_ids:
                    p = self.nodes[n]["pos"]
                    dist = self.distance2D(p, int_point)
                    if s not in nodes_dict:
                        nodes_dict[s] = {"node": n, "dist": dist}
                    else:
                        if dist < nodes_dict[s]["dist"]:
                            nodes_dict[s] = {"node": n, "dist": dist}

            if debug: print("nodes_dict: ", nodes_dict)
            
            nodes_comb = [v["node"] for k,v in nodes_dict.items()]
            combinations_list = list(map(list, itertools.combinations(nodes_comb, 2)))

            # check existance of path, otherwise add edge
            pred_edges = []
            for c0, c1 in combinations_list:
                pred_edges.append(tuple([c0, c1]))

            if debug: print("pred: ", pred_edges)

            overall_edges.extend(pred_edges)
        
        overall_edges = list(set(overall_edges))

        return overall_edges  



    def computeLocalPredictionEdgesExcluded(self, int_dict, debug=False):
        

        if debug: print("NEW INT POINTS: ", int_dict)

        # step 3: loop for each intersection
        overall_edges = []
        for k, v in int_dict.items():
            int_point = v["point"]
            seg_ids = v["segments"]

            nodes_dict = {}
            for n in self.single_nodes:
                s = self.nodes[n]["segment"]
                if s in seg_ids:
                    p = self.nodes[n]["pos"]
                    dist = self.distance2D(p, int_point)
                    if s not in nodes_dict:
                        nodes_dict[s] = {"node": n, "dist": dist}
                    else:
                        if dist < nodes_dict[s]["dist"]:
                            nodes_dict[s] = {"node": n, "dist": dist}

            if debug: print("nodes_dict: ", nodes_dict)
            
            nodes_comb = [v["node"] for k,v in nodes_dict.items()]
            combinations_list = list(map(list, itertools.combinations(nodes_comb, 2)))

            # check existance of path, otherwise add edge
            pred_edges = []
            for c0, c1 in combinations_list:
                pred_edges.append(tuple([c0, c1]))

            if debug: print("pred: ", pred_edges)

            overall_edges.extend(pred_edges)
        
        overall_edges = list(set(overall_edges))
        #print(overall_edges)

        return overall_edges  



    