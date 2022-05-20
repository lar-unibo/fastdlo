import torch
import os, sys
import numpy as np
import arrow

from fastdlo.siam_net.nn_models import SiameseNetwork
from fastdlo.siam_net.nn_dataset import AriadnePredictData

from termcolor import cprint 

class NN(object):

    def __init__(self, device, checkpoint_path):

        self.device = device 
        self.checkpoint_path = checkpoint_path
        self.model = SiameseNetwork(input_dim=7, hidden_dim=32, output_dim=16)

        # WEIGHTS LOADING
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.model.eval()


    def gaussian(self, val):
        return np.e**(-val**2)


    def predict(self, data, threshold=0.0):
        t0 = arrow.utcnow()

        out = []
        for sample in data:
            t1 = arrow.utcnow()

            z0 = self.model.forward_once(sample["feat_0"].unsqueeze(0))
            z1 = self.model.forward_once(sample["feat_1"].unsqueeze(0))
            prob = torch.exp(-torch.cdist(z0,z1)**2).squeeze().detach().cpu().numpy()

            # logging
            list_nodes = [sample["node_0"], sample["node_1"]]
            tot_time = (arrow.utcnow()-t1).total_seconds() * 1000
            #print(f"Nodes: {list_nodes}\t| Score: {prob:.4f} \t| Time: {tot_time:.4f} ms")

            if prob > threshold:
                out.append({"node_0": list_nodes[0], "node_1": list_nodes[1], "score": prob})


        #cprint(f"time NN prediction: {(arrow.utcnow()-t0).total_seconds() * 1000:.4f} ms", "yellow")
        return out


    def predictBatch(self, data, threshold=0.5, log=False):
        if len(data) == 0:
            return []

        t0 = arrow.utcnow()
        loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)

        out = []
        for batch in loader:
            z0 = self.model.forward_once(batch["feat_0"])
            z1 = self.model.forward_once(batch["feat_1"])

            for it, (zz0, zz1) in enumerate(zip(z0,z1)):
                prob = torch.exp(-torch.cdist(zz0.unsqueeze(0), zz1.unsqueeze(0))**2).squeeze().detach().cpu().numpy()

                if prob > threshold:
                    list_nodes = [int(batch["node_0"][it].detach().numpy()), int(batch["node_1"][it].detach().numpy())]
                    if log: print(f"Nodes: {list_nodes}\t| Score: {prob:.4f}")

                    out.append({"node_0": list_nodes[0], "node_1": list_nodes[1], "score": prob})

        if log: print(f"Tot Time: {(arrow.utcnow()-t0).total_seconds() * 1000:.4f} ms")
        return out

    

if __name__ == "__main__":

    # PARAMS
    DEVICE = "cpu"
    graph_name = "pred.pickle"

    script_path = os.path.dirname(os.path.realpath(__file__))

    MAIN_DATASET_PATH = os.path.join(os.path.split(script_path)[0], "data/dataset")
    test_folder = os.path.join(MAIN_DATASET_PATH, "test")
    
    
    
    sample_folder = os.path.join(test_folder, "806")
    sample_path = os.path.join(sample_folder, graph_name)

    CHECKPOINTS_PATH = os.path.join(script_path, "checkpoints")
    checkpoint_file = os.path.join(CHECKPOINTS_PATH, "best_nn.pth")

    # NN
    network = NN(device=DEVICE, checkpoint_path=checkpoint_file)

    # 
    data = AriadnePredictData.getAllPairsFromMemory(sample_path)   
    
    network.predictBatch(data)
    network.predict(data)   