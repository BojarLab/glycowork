import torch

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
