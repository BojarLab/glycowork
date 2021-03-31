import torch
from torch_geometric.nn import TopKPooling, GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from glycowork.glycan_data.loader import lib

class SweetNet(torch.nn.Module):
    def __init__(self, lib_size, num_classes = 1):
        super(SweetNet, self).__init__() 

        self.conv1 = GraphConv(128, 128)
        self.pool1 = TopKPooling(128, ratio = 0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio = 0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio = 0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings = lib_size+1, embedding_dim = 128)
        self.lin1 = torch.nn.Linear(256, 1024)
        self.lin2 = torch.nn.Linear(1024, 64)
        self.lin3 = torch.nn.Linear(64, num_classes)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.LeakyReLU()
        self.act2 = torch.nn.LeakyReLU()      
  
    def forward(self, x, edge_index, batch, inference = False):
        x = self.item_embedding(x)
        x = x.squeeze(1) 

        x = F.leaky_relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _, _= self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

        x = F.leaky_relu(self.conv2(x, edge_index))
     
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

        x = F.leaky_relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

        x = x1 + x2 + x3
        
        x = self.lin1(x)
        x = self.bn1(self.act1(x))
        x = self.lin2(x)
        x = self.bn2(self.act2(x))      
        x = F.dropout(x, p = 0.5, training = self.training)

        x = self.lin3(x).squeeze(1)

        if inference:
          x_out = x1 + x2 + x3
          return x, x_out
        else:
          return x

def init_weights(model, sparsity = 0.1):
    """initializes linear layers of PyTorch model with a sparse initialization\n
    model -- neural network (such as SweetNet) for analyzing glycans\n
    sparsity -- proportion of sparsity after initialization; default:0.1 / 10%
    """
    if type(model) == torch.nn.Linear:
        torch.nn.init.sparse_(model.weight, sparsity = sparsity)

def prep_model(model_type, num_classes, libr = None):
    """wrapper to instantiate model, initialize it, and put it on the GPU\n
    model_type -- string indicating the type of model\n
    num_classes -- number of unique classes for classification\n
    libr -- sorted list of unique glycoletters observed in the glycans of our dataset\n

    returns PyTorch model object
    """
    if libr is None:
        libr = lib
    if model_type == 'SweetNet':
        model = SweetNet(len(libr), num_classes = num_classes)
        model = model.apply(init_weights)
        model = model.cuda()
    else:
        print("Invalid Model Type")
    return model
