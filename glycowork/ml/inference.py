import pandas as pd
import numpy as np
import pkg_resources
import torch
from glycowork.glycan_data.loader import lib, unwrap
from glycowork.ml.processing import dataset_to_dataloader

try:
  from torch_geometric.data import Data
  from torch_geometric.loader import DataLoader
except ImportError:
  raise ImportError('<torch_geometric missing; cannot do deep learning>')

io = pkg_resources.resource_stream(__name__,
                                   "glycowork_lectinoracle_background_correction.csv")
df_corr = pd.read_csv(io)

#try:
#  import esm
#except ImportError:
#  print('<esm missing; cannot use get_esm1b_representations>')

def glycans_to_emb(glycans, model, libr = None, batch_size = 32, rep = True,
                   class_list = None):
    """Returns a dataframe of learned representations for a list of glycans\n
    | Arguments:
    | :-
    | glycans (list): list of glycans in IUPAC-condensed as strings
    | model (PyTorch object): trained graph neural network (such as SweetNet) for analyzing glycans
    | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset
    | batch_size (int): change to batch_size used during training; default:32
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

def get_multi_pred(prot, glycans, model, prot_dic,
                   background_correction = False, correction_df = None,
                   batch_size = 128, libr = None):
  """Inner function to actually get predictions for lectin-glycan binding from LectinOracle-type model\n
  | Arguments:
  | :-
  | prot (string): protein amino acid sequence
  | glycans (list): list of glycans in IUPACcondensed
  | model (PyTorch object): trained LectinOracle-type model
  | prot_dic (dictionary): dictionary of type protein sequence:ESM1b representation
  | background_correction (bool): whether to correct predictions for background; default:False
  | correction_df (dataframe): background prediction for (ideally) all provided glycans; default:None
  | batch_size (int): change to batch_size used during training; default:128
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset\n
  | Returns:
  | :-
  | Returns dataframe of glycan sequences and predicted binding to prot
  """
  if libr is None:
      libr = lib
  try:
    rep = prot_dic[prot]
  except:
    print('new protein, no stored embedding')
  train_loader = dataset_to_dataloader(glycans, [0.99]*len(glycans),
                                       libr = libr, batch_size = batch_size,
                                       shuffle = False, extra_feature = [rep]*len(glycans))
  model = model.eval()
  res = []
  for k in train_loader:
    x, y, edge_index, prot, batch = k.x, k.y, k.edge_index, k.train_idx, k.batch
    x = x.cuda()
    y = y.cuda()
    prot = prot.view(max(batch)+1, -1).cuda()
    edge_index = edge_index.cuda()
    batch = batch.cuda()
    pred = model(prot, x, edge_index, batch)
    res.append(pred)
  #res = unwrap([res[k].detach().cpu().numpy() for k in range(len(res))][0].tolist())
  res = unwrap([res[k].detach().cpu().numpy() for k in range(len(res))])
  res = [k.tolist()[0] for k in res]
  if background_correction:
    correction_df = pd.Series(correction_df.pred.values,
                              index = correction_df.motif).to_dict()
    bg_res = [correction_df[j] if j in list(correction_df.keys()) else 0 for j in glycans]
    if 0 in bg_res:
      print("Warning: not all glycans are in the correction_df; consider adding their background to correction_df")
    res = [a_i - b_i for a_i, b_i in zip(res, bg_res)]
  return res

def get_lectin_preds(prot, glycans, model, prot_dic, background_correction = False,
                     correction_df = None, batch_size = 128, libr = None, sort = True):
  """Wrapper that uses LectinOracle-type model for predicting binding of protein to glycans\n
  | Arguments:
  | :-
  | prot (string): protein amino acid sequence
  | glycans (list): list of glycans in IUPACcondensed
  | model (PyTorch object): trained LectinOracle-type model
  | prot_dic (dictionary): dictionary of type protein sequence:ESM1b representation
  | background_correction (bool): whether to correct predictions for background; default:False
  | correction_df (dataframe): background prediction for (ideally) all provided glycans; default:V4 correction file
  | batch_size (int): change to batch_size used during training; default:128
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset
  | sort (bool): whether to sort prediction results descendingly; default:True\n
  | Returns:
  | :-
  | Returns dataframe of glycan sequences and predicted binding to prot
  """
  if libr is None:
      libr = lib
  if correction_df is None:
    correction_df = df_corr
  preds = get_multi_pred(prot, glycans, model, prot_dic,
                         batch_size = batch_size, libr = libr)
  df_pred = pd.DataFrame(glycans, columns = ['motif'])
  df_pred['pred'] = preds
  if background_correction:
    correction_df = pd.Series(correction_df.pred.values,
                              index = correction_df.motif).to_dict()
    for j in df_pred.motif.values.tolist():
      motif_idx = df_pred.motif.values.tolist().index(j)
      try:
        df_pred.at[motif_idx, 'pred'] = df_pred.iloc[motif_idx, 1] - correction_df[j]
      except:
        pass
  if sort:
    df_pred.sort_values('pred', ascending = True, inplace = True)
  return df_pred

def get_esm1b_representations(prots, model, alphabet):
  """Retrieves ESM1b representations of protein for using them as input for LectinOracle\n
  | Arguments:
  | :-
  | prots (list): list of protein sequences (strings) that should be converted
  | model (ESM1b object): trained ESM1b model; from running esm.pretrained.esm1b_t33_650M_UR50S()
  | alphabet (ESM1b object): used for converting sequences; from running esm.pretrained.esm1b_t33_650M_UR50S()\n
  | Returns:
  | :-
  | Returns dictionary of the form protein sequence:ESM1b representation
  """
  #model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
  batch_converter = alphabet.get_batch_converter()
  prots = list(set(prots))
  data_list = []
  for k in range(0,len(prots)):
    if len(prots[k])<1000:
      data_list.append(('protein'+str(k), prots[k][:np.min([len(prots[k]),
                                                               1000])]))
    else:
      data_list.append(('protein'+str(k), prots[k][:1000]))
  batch_labels, batch_strs, batch_tokens = batch_converter(data_list)
  with torch.no_grad():
      results = model(batch_tokens,
                      repr_layers = [33], return_contacts = False)
  token_representations = results["representations"][33]
  sequence_representations = []
  for i, (_, seq) in enumerate(data_list):
      sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
  prot_dic =  {prots[k]:sequence_representations[k].tolist() for k in range(len(sequence_representations))}
  return prot_dic
