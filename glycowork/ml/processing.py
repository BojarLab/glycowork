from glycowork.motif.graph import *

try:
  from torch_geometric.data import Data, DataLoader
except ImportError:
    raise ImportError('<torch_geometric missing; cannot do deep learning>')

def dataset_to_graphs(glycan_list, labels, libr = lib, label_type = torch.long, separate = False,
                      context = False, error_catch = False, wo_labels = False):
  """wrapper function to convert a whole list of glycans into a graph dataset
  glycan_list -- list of IUPACcondensed glycan sequences (string)
  label_type -- which tensor type for label, default is torch.long for binary labels, change to torch.float for continuous
  separate -- True returns node list / edge list / label list as separate files; False returns list of data tuples; default is False
  lib -- sorted list of unique glycoletters observed in the glycans of our dataset
  context -- legacy-ish; used for generating graph context dataset for pre-training; keep at False
  error_catch -- troubleshooting option, True will print glycans that cannot be converted into graphs; default is False
  wo_labels -- change to True if you do not want to pass and receive labels; default is False

  returns list of node list / edge list / label list data tuples
  """
  if error_catch:
    glycan_graphs = []
    for k in glycan_list:
      try:
        glycan_graphs.append(glycan_to_graph(k, libr))
      except:
        print(k)
  else:
    glycan_graphs = [glycan_to_graph(k, libr) for k in glycan_list]
  if separate:
    glycan_nodes, glycan_edges = zip(*glycan_graphs)
    return list(glycan_nodes), list(glycan_edges), labels
  else:
    if context:
      contexts = [ggraph_to_context(k, lib = lib) for k in glycan_graphs]
      labels = [k[1] for k in contexts]
      labels = [item for sublist in labels for item in sublist]
      contexts = [k[0] for k in contexts]
      contexts = [item for sublist in contexts for item in sublist]
      data = [Data(x = torch.tensor(contexts[k][0], dtype = torch.long),
                 y = torch.tensor(labels[k], dtype = label_type),
                 edge_index = torch.tensor([contexts[k][1][0],contexts[k][1][1]], dtype = torch.long)) for k in range(len(contexts))]
      return data
    else:
      if wo_labels:
        glycan_nodes, glycan_edges = zip(*glycan_graphs)
        glycan_graphs = list(zip(glycan_nodes, glycan_edges))
        data = [Data(x = torch.tensor(k[0], dtype = torch.long),
                     edge_index = torch.tensor([k[1][0],k[1][1]], dtype = torch.long)) for k in glycan_graphs]
        return data
      else:
        glycan_nodes, glycan_edges = zip(*glycan_graphs)
        glycan_graphs = list(zip(glycan_nodes, glycan_edges, labels))
        data = [Data(x = torch.tensor(k[0], dtype = torch.long),
                 y = torch.tensor([k[2]], dtype = label_type),
                 edge_index = torch.tensor([k[1][0],k[1][1]], dtype = torch.long)) for k in glycan_graphs]
        return data
