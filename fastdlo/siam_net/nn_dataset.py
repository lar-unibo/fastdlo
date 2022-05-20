from codecs import encode
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import glob, os, sys
import random 
import math


class AriadnePredictData():

    @ staticmethod
    def buildNodeFeatures(nodes_dict, node_id):

        attr_id = nodes_dict[node_id]
               
        pos_d = np.array(attr_id["pos_d"])
        pos_d = (pos_d + 1) / 2

        dir = np.array(attr_id["pos_d"])
        angle = math.atan(dir[1]/dir[0])
        angle = (angle + math.pi*0.5) / math.pi

        vector = [  float(pos_d[0]), float(pos_d[1]), float(angle),
                    float(attr_id["radius"]),
                    float(attr_id["color"][0]), float(attr_id["color"][1]), float(attr_id["color"][2])]
        return vector


    @ staticmethod
    def getAllPairsFromMemory(path):

        with open(path, 'rb') as handle:
            graph = pickle.load(handle)
 
        nodes = graph["nodes"]
        pred_edges = graph["pred_edges"]

        return AriadnePredictData.makeTensors(nodes, pred_edges)

    @ staticmethod
    def makeTensors(nodes, pred_edges):
        rv = []
        for e in pred_edges:
            
            x0 = AriadnePredictData.buildNodeFeatures(nodes, node_id = e[0])
            x0 = torch.Tensor(x0)

            x1 = AriadnePredictData.buildNodeFeatures(nodes, node_id = e[1])
            x1 = torch.Tensor(x1)

            rv.append({ "node_0": e[0], "node_1": e[1], "feat_0": x0, "feat_1": x1})
        return rv

    @ staticmethod
    def getAllPairs(nodes, pred_edges):
        return AriadnePredictData.makeTensors(nodes, pred_edges)
  





class AriadneDataset(Dataset):

    def __init__(self, dataset_path, device, graph_name_pickle = "train.pickle"):
        
        self.graph_name_pickle = graph_name_pickle
        self.dataset_path = dataset_path
        self.device = device

        self.files = sorted(glob.glob(os.path.join(self.dataset_path, "*")))

        self.data = []
        for f in self.files:
            folder_id = os.path.basename(os.path.normpath(f))
            graph_path = os.path.join(f, self.graph_name_pickle)
            try:
                if os.path.exists(graph_path):
                    data_dict = self.retrieveData(graph_path)
                    triplet = self.getAllTriplet(data_dict, folder_id)
                    self.data.extend(triplet)
            except:
                pass


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


    def retrieveData(self, path):
        with open(path, 'rb') as handle:
            graph = pickle.load(handle)
 
        nodes = graph["nodes"]
        pos_edges_sup = graph["pos_edges_sup"]
        neg_edges_sup = graph["neg_edges_sup"]

        pos_edges = []
        for t0,t1 in pos_edges_sup:
            if tuple([t0,t1]) not in pos_edges and tuple([t1,t0]) not in pos_edges:
                pos_edges.append(tuple([t0,t1]))

        neg_edges = []
        for t0,t1 in neg_edges_sup:
            if tuple([t0,t1]) not in neg_edges and tuple([t1,t0]) not in neg_edges:
                neg_edges.append(tuple([t0,t1]))

        return {"nodes": nodes, "pos_edges": pos_edges, "neg_edges": neg_edges}
    


    def buildNodeFeatures(self, nodes_dict, node_id, node_2):

        attr_id = nodes_dict[node_id]
               
        pos_d = np.array(attr_id["pos_d"])
        pos_d = (pos_d + 1) / 2

        dir = np.array(attr_id["pos_d"])
        angle = math.atan(dir[1]/dir[0])
        angle = (angle + math.pi*0.5) / math.pi

        vector = [  float(pos_d[0]), float(pos_d[1]), float(angle),
                    float(attr_id["radius"]),
                    float(attr_id["color"][0]), float(attr_id["color"][1]), float(attr_id["color"][2])]
        return vector


    def getAllTriplet(self, data_dict, folder_id):

        rv = []
        for e in data_dict["pos_edges"]:
            
            x_target = self.buildNodeFeatures(data_dict["nodes"], node_id = e[0], node_2 = e[1])
            x_target = torch.Tensor(x_target)

            x_pos = self.buildNodeFeatures(data_dict["nodes"], node_id = e[1], node_2 = e[0])
            x_pos = torch.Tensor(x_pos)

            neg_of_target = []
            for p0, p1 in data_dict["neg_edges"]:
                if e[0] == p0:
                    neg_of_target.append(p1)
                elif e[0] == p1:
                    neg_of_target.append(p0)

            for neg_id in neg_of_target:
                x_neg = self.buildNodeFeatures(data_dict["nodes"], node_id=neg_id, node_2 = e[0])
                x_neg = torch.Tensor(x_neg)

                rv.append({ "folder_id": folder_id, "node_t": e[0], "node_p": e[1], "node_n": neg_id, "target": x_target, "pos": x_pos, "neg": x_neg})

        return rv




    def getRandomTriplet(self, data_dict, folder_id):

        e = random.choice(data_dict["pos_edges"])
        
        x_target = self.buildNodeFeatures(data_dict["nodes"], node_id = e[0], node_2 = e[1])
        x_target = torch.Tensor(x_target)

        x_pos = self.buildNodeFeatures(data_dict["nodes"], node_id = e[1], node_2 = e[0])
        x_pos = torch.Tensor(x_pos)

        neg_of_target = []
        for p0, p1 in data_dict["neg_edges"]:
            if e[0] == p0:
                neg_of_target.append(p1)
            elif e[0] == p1:
                neg_of_target.append(p0)


        neg_id = random.choice(neg_of_target)
        x_neg = self.buildNodeFeatures(data_dict["nodes"], node_id=neg_id, node_2 = e[0])
        x_neg = torch.Tensor(x_neg)
        
        rv =  { "folder_id": folder_id, "node_t": e[0], "node_p": e[1], "node_n": neg_id, "target": x_target, "pos": x_pos, "neg": x_neg}
        
        return [rv]