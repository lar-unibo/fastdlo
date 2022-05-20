import torch
import copy, os
import numpy as np
import arrow
from termcolor import cprint
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from nn_dataset import AriadneDataset
from nn_models import SiameseNetwork


class NN(object):


    def __init__(self, device, dataset_train_path, dataset_val_path, graph_name, checkpoints_path, 
                num_epochs = 250, lr=0.0005, bs=128, hidden_dim=32, out_dim=16):

        self.device = device 
        self.dataset_train_path = dataset_train_path
        self.dataset_val_path = dataset_val_path
        self.checkpoints_path = checkpoints_path
        cprint("Dataset train path: {}".format(self.dataset_train_path), "yellow")
        cprint("Dataset val path: {}".format(self.dataset_val_path), "yellow")

        self.graph_name = graph_name
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = bs

        # DATASET
        dataset_train = AriadneDataset(self.dataset_train_path, device=self.device, graph_name_pickle=self.graph_name)
        dataset_val = AriadneDataset(self.dataset_val_path, device=self.device, graph_name_pickle=self.graph_name)

        cprint("DATASET: total {}, train {}, val {}".format(len(dataset_train)+len(dataset_val), len(dataset_train), len(dataset_val)), "yellow")
        self.train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=0)

        # MODEL INIT
        num_features = dataset_train[0]["target"].size(0)

        self.model = SiameseNetwork(input_dim=num_features, hidden_dim=hidden_dim, output_dim=out_dim)
        self.criterion = torch.nn.TripletMarginLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        
        cprint("num epochs {}, lr {}, batch size {}".format(self.num_epochs, self.lr, self.batch_size), "yellow")
        cprint("num features {}, hidden dimension {}, out dimension {}".format(num_features, hidden_dim, out_dim), "yellow")


    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        t, p, n = self.model.forward(data["target"], data["pos"], data["neg"])

        loss = self.criterion(t, p, n)
        loss.backward()
        self.optimizer.step()
        return loss


    @torch.no_grad()
    def val(self, data):
        self.model.eval()
        t, p, n = self.model.forward(data["target"], data["pos"], data["neg"])
        val_loss = self.criterion(t, p, n)

        prob_tp = torch.exp(-torch.cdist(t,p)**2)
        prob_tn = torch.exp(-torch.cdist(t,n)**2)

        val_pred = torch.cat([prob_tp, prob_tn], dim=-1).squeeze()
        val_true = torch.cat([torch.ones(prob_tp.size(0)), torch.zeros(prob_tn.size(0))])
        val_auc = roc_auc_score(val_true, val_pred)

        return val_loss, val_auc


    def predict(self, model, loader):

        for batch in loader:

            t0 = arrow.utcnow()

            t, p, n = model.forward(batch["target"], batch["pos"], batch["neg"])

            prob_tp = torch.exp(-torch.cdist(t,p)**2).squeeze().detach().cpu().numpy()
            prob_tn = torch.exp(-torch.cdist(t,n)**2).squeeze().detach().cpu().numpy()

            list_nodes = [int(batch["node_t"].detach().numpy()), int(batch["node_p"].detach().numpy()), int(batch["node_n"].detach().numpy())]
            folder_id = batch["folder_id"]

            if prob_tp > prob_tn: result = u'\u2713'  
            else: result = u'\u2717'    

            tot_time = (arrow.utcnow()-t0).total_seconds() * 1000
            print(f"Folder: {folder_id}\t| Nodes (t,p,n) {list_nodes}\t| Scores: {prob_tp:.4f} -- {prob_tn:.4f}\t| Time: {tot_time:.4f} ms\t| Result: {result} ")



    def trainLoop(self):
        min_val_auc = 1000
        ep_best = 0
        with tqdm(total=self.num_epochs) as pbar:

            for ep in range(self.num_epochs):
                
                # TRAIN
                tot_loss = 0
                for batch in self.train_loader:
                    loss = self.train(batch)
                    tot_loss += loss.item()

                ep_loss = tot_loss/len(self.train_loader)

                # VAL
                tot_val_loss, tot_val_auc = 0.0, 0.0
                for batch in self.val_loader:
                    val_loss, val_auc = self.val(batch)
                    tot_val_loss += val_loss.item()
                    tot_val_auc += val_auc.item()

                ep_val_loss = tot_val_loss/len(self.val_loader)
                ep_val_auc = tot_val_auc/len(self.val_loader)

                if ep_val_loss < min_val_auc:
                    min_val_auc = ep_val_loss
                    self.best_model = copy.deepcopy(self.model)
                    ep_best = ep
            

                pbar.set_postfix(**{'loss': ep_loss, 'val_loss': ep_val_loss, 'val_auc': ep_val_auc})
                pbar.update()
        
        print(f'Best Model at epoch {ep_best}')

    def saveModel(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved to {}".format(path))

    def saveBestModel(self, path):
        torch.save(self.best_model.state_dict(), path)
        print("Model saved to {}".format(path))



if __name__ == "__main__":

    script_path = os.path.dirname(os.path.realpath(__file__))
    MAIN_DATASET_PATH = "/home/lar/dev/fast_dlo_segmentation/data/dataset_big"
    dataset_train = os.path.join(MAIN_DATASET_PATH, "train")
    dataset_val = os.path.join(MAIN_DATASET_PATH, "val")
    CHECKPOINTS_PATH = os.path.join(script_path, "checkpoints_big")
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
    DEVICE = "cpu"
    graph_name = "train.pickle"

    # PARAMS
    LR = 0.0005
    BS = 128
    EPOCHS = 50


    network = NN(   
                device=DEVICE, 
                dataset_train_path=dataset_train, 
                dataset_val_path=dataset_val, 
                graph_name=graph_name,
                checkpoints_path=CHECKPOINTS_PATH,
                num_epochs=EPOCHS,
                lr=LR,
                bs=BS
                )

    network.trainLoop()
    
    network.saveBestModel(path=os.path.join(CHECKPOINTS_PATH, "best_nn.pth"))


    # DATASET TEST
    test_set = AriadneDataset(dataset_path=dataset_val, device=DEVICE, graph_name_pickle=graph_name)   
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

    network.predict(model=network.best_model, loader=test_loader)