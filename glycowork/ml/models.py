from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Literal

import numpy as np
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GraphConv
    from torch_geometric.nn import global_mean_pool as gap
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
except ImportError:
  raise ImportError("<torch or torch_geometric missing; did you do 'pip install glycowork[ml]'?>")
from glycowork.glycan_data.loader import lib, download_model


class SweetNet(torch.nn.Module):
    def __init__(self, lib_size: int, # number of unique tokens for graph nodes
                 num_classes: int = 1, # number of output classes (>1 for multilabel)
                 hidden_dim: int = 128 # dimension of hidden layers
                ) -> None:
        "given glycan graphs as input, predicts properties via a graph neural network"
        super(SweetNet, self).__init__()
        # Convolution operations on the graph
        self.conv1 = GraphConv(hidden_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)

        # Node embedding
        self.item_embedding = torch.nn.Embedding(num_embeddings=lib_size+1, embedding_dim=hidden_dim)
        # Fully connected part
        self.lin1 = torch.nn.Linear(hidden_dim, 1024)
        self.lin2 = torch.nn.Linear(1024, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.act1 = torch.nn.LeakyReLU()
        self.act2 = torch.nn.LeakyReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                inference: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Getting node features
        x = self.item_embedding(x)
        x = x.squeeze(1)

        # Graph convolution operations
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))
        x = gap(x, batch)

        # Fully connected part
        x = self.act1(self.bn1(self.lin1(x)))
        x_out = self.bn2(self.lin2(x))
        x = F.dropout(self.act2(x_out), p = 0.5, training = self.training)

        x = self.lin3(x).squeeze(1)

        if inference:
          return x, x_out
        else:
          return x


class NSequonPred(torch.nn.Module):
    def __init__(self) -> None:
        "given an ESM1b representation of N and 20 AA up + downstream, predicts whether it's a sequon"
        super(NSequonPred, self).__init__()
        self.fc1 = torch.nn.Linear(1280, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 64)
        self.fc4 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      x = F.dropout(F.rrelu(self.bn1(self.fc1(x))), p = 0.2, training = self.training)
      x = F.dropout(F.rrelu(self.bn2(self.fc2(x))), p = 0.2, training = self.training)
      x = F.dropout(F.rrelu(self.bn3(self.fc3(x))), p = 0.1, training = self.training)
      x = self.fc4(x)
      return x


def sigmoid_range(x: torch.Tensor, # input tensor
                 low: float, # lower bound of range
                 high: float # upper bound of range
                ) -> torch.Tensor: # sigmoid transformed tensor in range (low,high)
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low


class SigmoidRange(torch.nn.Module):
    def __init__(self, low: float, # lower bound of range
                 high: float # upper bound of range
                ) -> None:
      "Sigmoid module with range `(low, high)`"
      super(SigmoidRange, self).__init__()
      self.low, self.high = low, high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sigmoid_range(x, self.low, self.high)


class LectinOracle(torch.nn.Module):
  def __init__(self, input_size_glyco: int, # number of unique tokens for graph nodes
                 hidden_size: int = 128, # layer size for graph convolutions
                 num_classes: int = 1, # number of output classes (>1 for multilabel)
                 data_min: float = -11.355, # minimum observed value in training data
                 data_max: float = 23.892, # maximum observed value in training data
                 input_size_prot: int = 1280 # dimensionality of protein representations
                ) -> None:
    "given glycan graphs and protein representations as input, predicts protein-glycan binding"
    super(LectinOracle, self).__init__()
    self.input_size_prot = input_size_prot
    self.input_size_glyco = input_size_glyco
    self.hidden_size = hidden_size
    self.num_classes = num_classes
    self.data_min = data_min
    self.data_max = data_max

    # Graph convolution operations for the glycan
    self.conv1 = GraphConv(self.hidden_size, self.hidden_size)
    self.conv2 = GraphConv(self.hidden_size, self.hidden_size)
    self.conv3 = GraphConv(self.hidden_size, self.hidden_size)
    # Node embedding for the glycan
    self.item_embedding = torch.nn.Embedding(num_embeddings = self.input_size_glyco+1,
                                             embedding_dim = self.hidden_size)

    # Fully connected part for the protein
    self.prot_encoder1 = torch.nn.Linear(self.input_size_prot, 400)
    self.prot_encoder2 = torch.nn.Linear(400, 128)
    self.bn_prot1 = torch.nn.BatchNorm1d(400)
    self.bn_prot2 = torch.nn.BatchNorm1d(128)
    self.dp_prot1 = torch.nn.Dropout(0.2)
    self.dp_prot2 = torch.nn.Dropout(0.1)
    self.act_prot1 = torch.nn.LeakyReLU()
    self.act_prot2 = torch.nn.LeakyReLU()

    # Combined fully connected part
    self.fc1 = torch.nn.Linear(128+self.hidden_size, int(np.round(self.hidden_size/2)))
    self.fc2 = torch.nn.Linear(int(np.round(self.hidden_size/2)), self.num_classes)
    self.bn1 = torch.nn.BatchNorm1d(int(np.round(self.hidden_size/2)))
    self.dp1 = torch.nn.Dropout(0.5)
    self.act1 = torch.nn.LeakyReLU()
    self.sigmoid = SigmoidRange(self.data_min, self.data_max)

  def forward(self, prot: torch.Tensor, nodes: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
              inference: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # Fully connected part for the protein
    embedded_prot = self.bn_prot1(self.act_prot1(self.dp_prot1(self.prot_encoder1(prot))))
    embedded_prot = self.bn_prot2(self.act_prot2(self.dp_prot2(self.prot_encoder2(embedded_prot))))

    # Getting glycan node features
    x = self.item_embedding(nodes)
    x = x.squeeze(1)

    # Glycan graph convolution operations
    x = F.leaky_relu(self.conv1(x, edge_index))
    x = F.leaky_relu(self.conv2(x, edge_index))
    x = F.leaky_relu(self.conv3(x, edge_index))
    x = gap(x, batch)

    # Combining results from protein and glycan
    h_n = torch.cat((embedded_prot, x), 1)

    # Fully connected part
    h_n = self.act1(self.bn1(self.fc1(h_n)))

    x1 = self.fc2(self.dp1(h_n))
    x2 = self.fc2(self.dp1(h_n))
    x3 = self.fc2(self.dp1(h_n))
    x4 = self.fc2(self.dp1(h_n))
    x5 = self.fc2(self.dp1(h_n))
    x6 = self.fc2(self.dp1(h_n))
    x7 = self.fc2(self.dp1(h_n))
    x8 = self.fc2(self.dp1(h_n))

    out = self.sigmoid(torch.mean(torch.stack([x1, x2, x3, x4, x5, x6, x7, x8]), dim = 0))

    if inference:
      return out, embedded_prot, x
    else:
      return out


class LectinOracle_flex(torch.nn.Module):
  def __init__(self, input_size_glyco: int, # number of unique tokens for graph nodes
                 hidden_size: int = 128, # layer size for graph convolutions
                 num_classes: int = 1, # number of output classes (>1 for multilabel)
                 data_min: float = -11.355, # minimum observed value in training data
                 data_max: float = 23.892, # maximum observed value in training data
                 input_size_prot: int = 1000 # maximum protein sequence length for padding/cutting
                ) -> None:
    "given glycan graphs and protein sequences as input, predicts protein-glycan binding"
    super(LectinOracle_flex, self).__init__()
    self.input_size_prot = input_size_prot
    self.input_size_glyco = input_size_glyco
    self.hidden_size = hidden_size
    self.num_classes = num_classes
    self.data_min = data_min
    self.data_max = data_max

    # Graph convolution operations for the glycan
    self.conv1 = GraphConv(self.hidden_size, self.hidden_size)
    self.conv2 = GraphConv(self.hidden_size, self.hidden_size)
    self.conv3 = GraphConv(self.hidden_size, self.hidden_size)
    # Node embedding for the glycan
    self.item_embedding = torch.nn.Embedding(num_embeddings = self.input_size_glyco+1,
                                             embedding_dim = self.hidden_size)

    # ESM-1b mimicking
    self.fc1 = torch.nn.Linear(self.input_size_prot, 4000)
    self.fc2 = torch.nn.Linear(4000, 2000)
    self.fc3 = torch.nn.Linear(2000, 1280)
    self.dp1 = torch.nn.Dropout(0.3)
    self.dp2 = torch.nn.Dropout(0.2)
    self.act1 = torch.nn.LeakyReLU()
    self.act2 = torch.nn.LeakyReLU()
    self.bn1 = torch.nn.BatchNorm1d(4000)
    self.bn2 = torch.nn.BatchNorm1d(2000)

    # Fully connected part for the protein
    self.prot_encoder1 = torch.nn.Linear(1280, 400)
    self.prot_encoder2 = torch.nn.Linear(400, 128)
    self.dp_prot1 = torch.nn.Dropout(0.2)
    self.dp_prot2 = torch.nn.Dropout(0.1)
    self.bn_prot1 = torch.nn.BatchNorm1d(400)
    self.bn_prot2 = torch.nn.BatchNorm1d(128)
    self.act_prot1 = torch.nn.LeakyReLU()
    self.act_prot2 = torch.nn.LeakyReLU()

    # Combined fully connected part
    self.dp1_n = torch.nn.Dropout(0.5)
    self.fc1_n = torch.nn.Linear(128+self.hidden_size, int(np.round(self.hidden_size/2)))
    self.fc2_n = torch.nn.Linear(int(np.round(self.hidden_size/2)), self.num_classes)
    self.bn1_n = torch.nn.BatchNorm1d(int(np.round(self.hidden_size/2)))
    self.act1_n = torch.nn.LeakyReLU()
    self.sigmoid = SigmoidRange(self.data_min, self.data_max)

  def forward(self, prot: torch.Tensor, nodes: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
              inference: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # ESM-1b mimicking
    prot = self.dp1(self.act1(self.bn1(self.fc1(prot))))
    prot = self.dp2(self.act2(self.bn2(self.fc2(prot))))
    prot = self.fc3(prot)
    # Fully connected part for the protein
    embedded_prot = self.dp_prot1(self.act_prot1(self.bn_prot1(self.prot_encoder1(prot))))
    embedded_prot = self.dp_prot2(self.act_prot2(self.bn_prot2(self.prot_encoder2(embedded_prot))))
    # Getting glycan node features
    x = self.item_embedding(nodes)
    x = x.squeeze(1)
    # Glycan graph convolution operations
    x = F.leaky_relu(self.conv1(x, edge_index))
    x = F.leaky_relu(self.conv2(x, edge_index))
    x = F.leaky_relu(self.conv3(x, edge_index))
    x = gap(x, batch)
    # Combining results from protein and glycan
    h_n = torch.cat((embedded_prot, x), 1)
    # Fully connected part
    h_n = self.act1_n(self.bn1_n(self.fc1_n(h_n)))
    x1 = self.fc2_n(self.dp1(h_n))
    x2 = self.fc2_n(self.dp1(h_n))
    x3 = self.fc2_n(self.dp1(h_n))
    x4 = self.fc2_n(self.dp1(h_n))
    x5 = self.fc2_n(self.dp1(h_n))
    x6 = self.fc2_n(self.dp1(h_n))
    x7 = self.fc2_n(self.dp1(h_n))
    x8 = self.fc2_n(self.dp1(h_n))
    out = self.sigmoid(torch.mean(torch.stack([x1, x2, x3, x4, x5, x6, x7, x8]), dim = 0))
    if inference:
      return out, embedded_prot, x
    else:
      return out


def init_weights(model: torch.nn.Module, # neural network for analyzing glycans
                mode: str = 'sparse', # initialization algorithm: 'sparse', 'kaiming', 'xavier'
                sparsity: float = 0.1 # proportion of sparsity after initialization
               ) -> None:
    "initializes linear layers of PyTorch model with a weight initialization"
    if isinstance(model, torch.nn.Linear):
        if mode == 'sparse':
            torch.nn.init.sparse_(model.weight, sparsity = sparsity)
        elif mode == 'kaiming':
            torch.nn.init.kaiming_uniform_(model.weight)
        elif mode == 'xavier':
            torch.nn.init.xavier_uniform_(model.weight)
        else:
            print("This initialization option is not supported.")


def prep_model(model_type: Literal["SweetNet", "LectinOracle", "LectinOracle_flex", "NSequonPred"], # type of model to create
              num_classes: int, # number of unique classes for classification
              libr: Optional[Dict[str, int]] = None, # dictionary of form glycoletter:index
              trained: bool = False, # whether to use pretrained model
              hidden_dim: int = 128 # hidden dimension for the model (SweetNet/LectinOracle only)
             ) -> torch.nn.Module: # initialized PyTorch model
    "wrapper to instantiate model, initialize it, and put it on the GPU"
    if libr is None:
      libr = lib
    if model_type == 'SweetNet':
      model = SweetNet(len(libr), num_classes = num_classes, hidden_dim = hidden_dim)
      model = model.apply(lambda module: init_weights(module, mode = 'sparse'))
      if trained:
        if hidden_dim != 128:
          raise ValueError("Hidden dimension must be 128 for pretrained model")
        if not Path("SweetNet_v1_4.pt").exists():
          download_model("https://drive.google.com/file/d/1arIT31FpA1FCKSDVUuntc9-UQEUpcXVz/view?usp=sharing", local_path = "SweetNet_v1_4.pt")
        model.load_state_dict(torch.load("SweetNet_v1_4.pt", map_location = device, weights_only = True))
      model = model.to(device)
    elif model_type == 'LectinOracle':
      model = LectinOracle(len(libr), num_classes = num_classes, input_size_prot = int(10*hidden_dim))
      model = model.apply(lambda module: init_weights(module, mode = 'xavier'))
      if trained:
        if not Path("LectinOracle_v1_4.pt").exists():
          download_model("https://drive.google.com/file/d/1g5GnwJvGW0Zis2zwxjRsZE9t-ueV6-cP/view?usp=sharing", local_path = "LectinOracle_v1_4.pt")
        model.load_state_dict(torch.load("LectinOracle_v1_4.pt", map_location = device, weights_only = True))
      model = model.to(device)
    elif model_type == 'LectinOracle_flex':
      model = LectinOracle_flex(len(libr), num_classes = num_classes)
      model = model.apply(lambda module: init_weights(module, mode = 'xavier'))
      if trained:
        if not Path("LectinOracle_flex_v1_4.pt").exists():
          download_model("https://drive.google.com/file/d/1h051ql_LTfzQjuTpDzTrAqNTAPDpKqwB/view?usp=sharing", local_path = "LectinOracle_flex_v1_4.pt")
        model.load_state_dict(torch.load("LectinOracle_flex_v1_4.pt", map_location = device, weights_only = True))
      model = model.to(device)
    elif model_type == 'NSequonPred':
      model = NSequonPred()
      model = model.apply(lambda module: init_weights(module, mode = 'xavier'))
      if trained:
        if not Path("NSequonPred_v1_4.pt").exists():
          download_model("https://drive.google.com/file/d/12KQOfwCAUkXwCKw5DHYTh3uHEjXsJzSA/view?usp=sharing", local_path = "NSequonPred_v1_4.pt")
        model.load_state_dict(torch.load("NSequonPred_v1_4.pt", map_location = device, weights_only = True))
      model = model.to(device)
    else:
      print("Invalid Model Type")
    return model
