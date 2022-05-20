from torch import nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SiameseNetwork, self).__init__()
        
        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)

        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim)

        self.linear3 = nn.Linear(in_features=hidden_dim, out_features=output_dim)


    def forward_once(self, x):
        y = F.relu(self.bn1(self.linear1(x)))
        y = F.relu(self.bn2(self.linear2(y)))
        return F.normalize(self.linear3(y))


    def forward(self, input_t, input_p, input_n):
        output_t = self.forward_once(input_t)
        output_p = self.forward_once(input_p)
        output_n = self.forward_once(input_n)

        return output_t, output_p, output_n