import torch
import os
import numpy as np
from termcolor import cprint
from sklearn.metrics import roc_auc_score

from nn_dataset import AriadneDataset
from nn_models import SiameseNetwork


from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial


script_path = os.path.dirname(os.path.realpath(__file__))
MAIN_DATASET_PATH = os.path.join(os.path.split(script_path)[0], "data/dataset")
DATASET_TRAIN_PATH = os.path.join(MAIN_DATASET_PATH, "train")
DATASET_VAL_PATH = os.path.join(MAIN_DATASET_PATH, "val")
GRAPH_NAME = "train.pickle"

DEVICE = "cpu"
NUM_EPOCHS = 50

# DATASET
dataset_train = AriadneDataset(DATASET_TRAIN_PATH, device=DEVICE, graph_name_pickle=GRAPH_NAME)
dataset_val = AriadneDataset(DATASET_VAL_PATH, device=DEVICE, graph_name_pickle=GRAPH_NAME)
num_features = dataset_train[0]["target"].size(0)


def trainLoop(config, checkpoint_dir=None):

    print("Total: ", len(dataset_train)+len(dataset_val))
    print("Train: ", len(dataset_train))
    print("Val: ", len(dataset_val))
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=0)
    

    # MODEL INIT
    hidden_dim = config["hidden_dim"]
    out_dim = config["out_dim"]
    print("num features {}, hidden dimension {}, out dimension {}".format(num_features, hidden_dim, out_dim))

    model = SiameseNetwork(input_dim=num_features, hidden_dim=hidden_dim, output_dim=out_dim)
    criterion = torch.nn.TripletMarginWithDistanceLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"])
    
    cprint("num epochs {}, lr {}, batch size {}".format(NUM_EPOCHS, config["lr"], config["batch_size"]), "yellow")
    cprint("num features {}, hidden dimension {}, out dimension {}".format(num_features, hidden_dim, out_dim), "yellow")


    for ep in range(NUM_EPOCHS):

        # TRAIN
        model.train()
        tot_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            t, p, n = model.forward(data["target"], data["pos"], data["neg"])
            loss = criterion(t, p, n)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

        ep_loss = tot_loss/len(train_loader)

        # VAL
        tot_auc = 0.0
        val_loss = 0.0
        with torch.no_grad():
            model.eval()
            for data in val_loader:
                t, p, n = model.forward(data["target"], data["pos"], data["neg"])
                loss = criterion(t, p, n)

                prob_tp = torch.exp(-torch.cdist(t,p)**2)
                prob_tn = torch.exp(-torch.cdist(t,n)**2)

                val_pred = torch.cat([prob_tp, prob_tn], dim=-1).squeeze()
                val_true = torch.cat([torch.ones(prob_tp.size(0)), torch.zeros(prob_tn.size(0))])
                val_auc = roc_auc_score(val_true, val_pred)

                val_loss += loss.cpu().numpy()
                tot_auc += val_auc.item()

        ep_auc = tot_auc/len(val_loader)
        val_loss = val_loss/len(val_loader)


    with tune.checkpoint_dir(ep) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save((model.state_dict(), optimizer.state_dict()), path)

    tune.report(loss=val_loss, accuracy=ep_auc)

    print("Finished Training")
        


def test_accuracy(net):
    with torch.no_grad():
        net.eval()
        tot_auc = 0.0
        loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=0)
        for data in loader:
            t, p, n = net.forward(data["target"], data["pos"], data["neg"])

            prob_tp = torch.exp(-torch.cdist(t,p)**2)
            prob_tn = torch.exp(-torch.cdist(t,n)**2)

            val_pred = torch.cat([prob_tp, prob_tn], dim=-1).squeeze()
            val_true = torch.cat([torch.ones(prob_tp.size(0)), torch.zeros(prob_tn.size(0))])
            val_auc = roc_auc_score(val_true, val_pred)
            tot_auc += val_auc.item()

    return tot_auc/len(loader)



def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0):
    config = {
        "hidden_dim": tune.sample_from(lambda _: 2 ** np.random.randint(4, 9)),
        "out_dim": tune.sample_from(lambda _: 2 ** np.random.randint(4, 9)),
        "lr": tune.loguniform(1e-4, 1e-3),
        "batch_size": tune.choice([32, 64, 128, 256])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        partial(trainLoop),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        fail_fast="raise")

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    best_trained_model = SiameseNetwork(input_dim=num_features, hidden_dim=best_trial.config["hidden_dim"], output_dim=best_trial.config["out_dim"])
    best_trained_model.to(DEVICE)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model)
    print("Best trial val set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    main(num_samples=2, max_num_epochs=NUM_EPOCHS, gpus_per_trial=0)

