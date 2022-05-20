import os, pickle, cv2
from numpy import int_
from termcolor import cprint
import arrow
import numpy as np 

import fastdlo.proc.utils as utils
from fastdlo.proc.graphx import Graph
from fastdlo.proc.skel import Processing

import matplotlib.pyplot as plt

class LabelsPred():

    def computeExcluded(self, int_excluded_dict):
        return self.graph.computeLocalPredictionEdgesExcluded(int_excluded_dict)


    def compute(self, mask_img, source_img, mask_threshold=127, density=7, timings=True):


        t0 = arrow.utcnow()
        # skeleton
        mask = mask_img.copy()
        mask[mask_img <= mask_threshold] = 0
        mask[mask_img > mask_threshold] = 1
        skel = Processing(mask, density=density, drop_endpoints=True)    
        paths = skel.paths
        int_points_dict = skel.int_points
        dist_img = skel.dist_img

        
        if False:
            canvas = mask_img.copy()
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
            plt.imshow(canvas)
            for k, p in skel.paths.items():
                plt.plot(p[:,1], p[:,0], label=str(k))
            plt.legend()
            plt.show()


        skel_time = (arrow.utcnow() - t0).total_seconds() * 1000
        if timings: cprint("skel time: {0:.4f}".format(skel_time), "yellow")


        t1 = arrow.utcnow()
        # discretize skeleton
        vertices_dict, edges_dict, excluded_paths = utils.samplePaths(paths, density=density)


        if timings: cprint("discretize time: {0:.4f}".format((arrow.utcnow() - t1).total_seconds() * 1000), "yellow")
        

        t2 = arrow.utcnow()
        # attributes
        radii_dict = utils.computeRadii(vertices_dict, dist_img)
        if timings: cprint("attributes radii time: {0:.4f}".format((arrow.utcnow() - t2).total_seconds() * 1000), "yellow")

        #t22 = arrow.utcnow()
        #splines_dict = utils.computeSplines(vertices_dict, s=1e-2)
        #if timings: cprint("attributes spline time: {0:.4f}".format((arrow.utcnow() - t22).total_seconds() * 1000), "yellow")

        t3 = arrow.utcnow()
        # generate graph
        self.graph = Graph(vertices=vertices_dict, edges=edges_dict, radius=radii_dict, image=source_img, dist_img=dist_img)
        if timings: cprint("graph time: {0:.4f}".format((arrow.utcnow() - t3).total_seconds() * 1000), "yellow")

        # prediction edges
        t5 = arrow.utcnow()
        pred_edges = self.graph.computeLocalPredictionEdges(int_points_dict, excluded_paths)
        if timings: cprint("prediction edges time: {0:.4f}".format((arrow.utcnow() - t5).total_seconds() * 1000), "yellow")


        if timings: cprint("tot funct time: {0:.4f}".format((arrow.utcnow() - t0).total_seconds() * 1000), "yellow")

        return {"nodes": self.graph.nodes, "single_nodes": self.graph.single_nodes, "pred_edges": pred_edges, 
                "vertices": self.graph.vertices_dict_nodes, "radius": radii_dict, "intersections": int_points_dict, "time": {"skel": skel_time}}


    @ staticmethod
    def genPickle(graph, pred_edges):

        graph_dict = {}
        graph_dict["nodes"] = graph.getNodes()
        graph_dict["edges"] = graph.getMessageEdges()
        graph_dict["pred_edges"] = pred_edges
            
        return graph_dict

    @ staticmethod
    def savePickle(graph_dict, filepath):

        with open(filepath, 'wb') as handle:
            pickle.dump(graph_dict, handle) 

        print("Saved to ---> {}".format(filepath))


    @ staticmethod
    def main(dataset_path, split, save_graph=False, show_graph=False, save_pickle=False):
        
        split_path = os.path.join(dataset_path, split)

        folders = [int(f) for f in os.listdir(split_path) if not os.path.isfile(os.path.join(split_path, f))]

        for folder in sorted(folders):

            print("")
            cprint("FOLDER: {}".format(folder), "green")

            folder_path = os.path.join(split_path, str(folder))

            #try:
            #    os.remove(os.path.join(folder_path, "pred.pickle"))
            #except:
            #    pass

            t0 = arrow.utcnow()

            # COLOR
            source_img = cv2.imread(os.path.join(folder_path, "color.png"), cv2.IMREAD_COLOR)

            # MASK
            mask_img = cv2.imread(os.path.join(folder_path, "instance_3.png"), cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.bitwise_not(mask_img)

            rv = LabelsPred.compute(source_img=source_img, mask_img=mask_img, timings=False)
            graph, pred_edges, vertices_dict, ints_dict, radii_dict = rv["graph"], rv["pred_edges"], rv["vertices"], rv["intersections"], rv["radii"]
            
            graph_to_save = LabelsPred.genPickle(graph, pred_edges)

            cprint("TOT TIME: {0:.4f}".format((arrow.utcnow() - t0).total_seconds() * 1000), "yellow")
                
            if save_graph:
                graph.show(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB), savepath=os.path.join(folder_path, "graph.png"))
            elif show_graph:
                graph.show(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
            else:
                pass

            if save_pickle:
                filepath_save=os.path.join(folder_path, "pred.pickle")
                LabelsPred.savePickle(graph_to_save, filepath_save)


