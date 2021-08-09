import pandas as pd
import numpy as np
import torch
from glycowork.glycan_data.loader import lib
from glycowork.ml.processing import dataset_to_dataloader

try:
  from torch_geometric.data import Data, DataLoader
except ImportError:
    raise ImportError('<torch_geometric missing; cannot do deep learning>')

def glycans_to_emb(glycans, model, libr = None, batch_size = 32, rep = True,
                   class_list = None):
    """returns a dataframe of learned representations for a list of glycans\n
    | Arguments:
    | :-
    | glycans (list): list of glycans in IUPAC-condensed as strings
    | model (PyTorch object): trained graph neural network (such as SweetNet) for analyzing glycans
    | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset
    | batch_size (int): change to batch_size used during training; default is 32
    | rep (bool): True returns representations, False returns actual predicted labels; default is True
    | class_list (list): list of unique classes to map predictions\n
    | Returns:
    | :-
    | Returns dataframe of learned representations (columns) for each glycan (rows)
    """
    if libr is None:
      libr = lib
    glycan_loader = dataset_to_dataloader(glycans, range(len(glycans)),
                                          libr = libr, batch_size = batch_size,
                                          shuffle = False)
    res = []
    for data in glycan_loader:
        x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
        x = x.cuda()
        y = y.cuda()
        edge_index = edge_index.cuda()
        batch = batch.cuda()
        model = model.eval()
        pred, out = model(x, edge_index, batch, inference = True)
        if rep:
            res.append(out)
        else:
            res.append(pred)
    res2 = [res[k].detach().cpu().numpy() for k in range(len(res))]
    res2 = pd.DataFrame(np.concatenate(res2))
    if rep:
      return res2
    else:
      idx = res2.idxmax(axis = "columns").values.tolist()
      preds = [class_list[k] for k in idx]
      return preds
