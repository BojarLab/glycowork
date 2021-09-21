import os
import torch
from torch_geometric.nn import TopKPooling, GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from glycowork.glycan_data.loader import lib

this_dir, this_filename = os.path.split(__file__)  # Get path
data_path = os.path.join(this_dir, 'glycowork_sweetnet_species.pt')
SweetNet = torch.load(data_path)
data_path = os.path.join(this_dir, 'glycowork_lectinoracle_565.pt')
LectinOracle = torch.load(data_path)

class SweetNet(torch.nn.Module):
    def __init__(self, lib_size, num_classes = 1):
        super(SweetNet, self).__init__() 

        self.conv1 = GraphConv(128, 128)
        self.pool1 = TopKPooling(128, ratio = 0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio = 0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio = 0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings = lib_size+1,
                                                 embedding_dim = 128)
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

def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low

class SigmoidRange(torch.nn.Module):
    "Sigmoid module with range `(low, x_max)`"
    def __init__(self, low, high):
      super(SigmoidRange, self).__init__()
      self.low, self.high = low,high
    def forward(self, x): return sigmoid_range(x, self.low, self.high)

class LectinOracle(torch.nn.Module):
  def __init__(self, input_size_glyco, hidden_size = 128, num_classes = 1, data_min = -11.355,
               data_max = 23.892, input_size_prot = 1280):
    super(LectinOracle,self).__init__()
    self.input_size_prot = input_size_prot
    self.input_size_glyco = input_size_glyco
    self.hidden_size = hidden_size
    self.num_classes = num_classes
    self.data_min = data_min
    self.data_max = data_max
    
    self.conv1 = GraphConv(self.hidden_size, self.hidden_size)
    self.pool1 = TopKPooling(self.hidden_size, ratio = 0.8)
    self.conv2 = GraphConv(self.hidden_size, self.hidden_size)
    self.pool2 = TopKPooling(self.hidden_size, ratio = 0.8)
    self.conv3 = GraphConv(self.hidden_size, self.hidden_size)
    self.pool3 = TopKPooling(self.hidden_size, ratio = 0.8)
    self.item_embedding = torch.nn.Embedding(num_embeddings = self.input_size_glyco,
                                             embedding_dim = self.hidden_size)
    self.prot_encoder1 = torch.nn.Linear(self.input_size_prot, 400)
    self.prot_encoder2 = torch.nn.Linear(400, 128)
    self.dp1 = torch.nn.Dropout(0.5)
    self.dp_prot1 = torch.nn.Dropout(0.2)
    self.dp_prot2 = torch.nn.Dropout(0.1)
    self.fc1 = torch.nn.Linear(128+2*self.hidden_size, int(np.round(self.hidden_size/2)))
    self.fc2 = torch.nn.Linear(int(np.round(self.hidden_size/2)), self.num_classes)
    self.bn1 = torch.nn.BatchNorm1d(int(np.round(self.hidden_size/2)))
    self.bn_prot1 = torch.nn.BatchNorm1d(400)
    self.bn_prot2 = torch.nn.BatchNorm1d(128)
    self.act1 = torch.nn.LeakyReLU()
    self.act_prot1 = torch.nn.LeakyReLU()
    self.act_prot2 = torch.nn.LeakyReLU()
    self.sigmoid = SigmoidRange(self.data_min, self.data_max)
    
    
  def forward(self, prot, nodes, edge_index, batch, inference = False):
    embedded_prot = self.bn_prot1(self.act_prot1(self.dp_prot1(self.prot_encoder1(prot))))
    embedded_prot = self.bn_prot2(self.act_prot2(self.dp_prot2(self.prot_encoder2(embedded_prot))))

    x = self.item_embedding(nodes)
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

    h_n = torch.cat((embedded_prot, x), 1)
    
    h_n = self.bn1(self.act1(self.fc1(h_n)))

    #1
    x1 = self.fc2(self.dp1(h_n))
    #2
    x2 = self.fc2(self.dp1(h_n))
    #3
    x3 = self.fc2(self.dp1(h_n))
    #4
    x4 = self.fc2(self.dp1(h_n))
    #5
    x5 = self.fc2(self.dp1(h_n))
    #6
    x6 = self.fc2(self.dp1(h_n))
    #7
    x7 = self.fc2(self.dp1(h_n))
    #8
    x8 = self.fc2(self.dp1(h_n))
    
    out =  self.sigmoid(torch.mean(torch.stack([x1, x2, x3, x4, x5, x6, x7, x8]), dim = 0))
    
    if inference:
      return out, embedded_prot, x
    else:
      return out

def init_weights(model, sparsity = 0.1):
    """initializes linear layers of PyTorch model with a sparse initialization\n
    | Arguments:
    | :-
    | model (Pytorch object): neural network (such as SweetNet) for analyzing glycans
    | sparsity (float): proportion of sparsity after initialization; default:0.1 / 10%
    """
    if type(model) == torch.nn.Linear:
        torch.nn.init.sparse_(model.weight, sparsity = sparsity)

def prep_model(model_type, num_classes, libr = None,
               trained = False):
    """wrapper to instantiate model, initialize it, and put it on the GPU\n
    | Arguments:
    | :-
    | model_type (string): string indicating the type of model
    | num_classes (int): number of unique classes for classification
    | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset\n
    | Returns:
    | :-
    | Returns PyTorch model object
    """
    if libr is None:
        libr = lib
    if model_type == 'SweetNet':
        model = SweetNet(len(libr), num_classes = num_classes)
        model = model.apply(init_weights)
        if trained:
            model.load_state_dict(SweetNet)
        model = model.cuda()
    elif model_type == 'LectinOracle':
        model = LectinOracle(len(libr), num_classes = num_classes)
        model = model.apply(init_weights)
        if trained:
            model.load_state_dict(LectinOracle)
        model = model.cuda()
    else:
        print("Invalid Model Type")
    return model
