import pandas as pd
import numpy as np
from importlib import resources
from typing import Any
import math
try:
    import torch
    from torch.utils.data import Dataset
    # Choosing the right computing architecture
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
except ImportError:
  raise ImportError("<torch missing; did you do 'pip install glycowork[ml]'?>")
from glycowork.glycan_data.loader import lib, unwrap
from glycowork.motif.tokenization import prot_to_coded
from glycowork.ml.processing import dataset_to_dataloader


class SimpleDataset(Dataset):
  def __init__(self, x: list[float], y: list[float]) -> None:
    self.x = x
    self.y = y

  def __len__(self) -> int:
    return len(self.x)

  def __getitem__(self, index: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    inp = self.x[index]
    out = self.y[index]
    return torch.FloatTensor(inp), torch.FloatTensor([out])


def sigmoid(x: float) -> float:
  return 1 / (1 + math.exp(-x))


def glycans_to_emb(glycans: list[str],  model: torch.nn.Module,  libr: dict[str, int] | None = None,  batch_size: int = 32, 
                   rep: bool = True, class_list: list[str] | None = None) -> pd.DataFrame | list[str]:
    """Returns a dataframe of learned representations for a list of glycans\n
    | Arguments:
    | :-
    | glycans (list): list of glycans in IUPAC-condensed as strings
    | model (PyTorch object): trained graph neural network (such as SweetNet) for analyzing glycans
    | libr (dict): dictionary of form glycoletter:index
    | batch_size (int): change to batch_size used during training; default:32
    | rep (bool): True returns representations, False returns actual predicted labels; default is True
    | class_list (list): list of unique classes to map predictions\n
    | Returns:
    | :-
    | Returns dataframe of learned representations (columns) for each glycan (rows)"""
    if libr is None:
      libr = lib
    # Preparing dataset for PyTorch
    glycan_loader = dataset_to_dataloader(glycans, range(len(glycans)),
                                          libr = libr, batch_size = batch_size,
                                          shuffle = False)
    res = []
    model = model.eval()
    # Get predictions for each mini-batch
    for data in glycan_loader:
        x, y, edge_index, batch = data.labels, data.y, data.edge_index, data.batch
        x = x.to(device)
        y = y.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        pred, out = model(x, edge_index, batch, inference = True)
        # Unpacking and combining predictions
        if rep:
            res.extend(out.detach().cpu().numpy())
        else:
            res.extend(pred.detach().cpu().numpy())
    if rep:
        return pd.DataFrame(res)
    else:
        preds = [class_list[k] for k in np.argmax(res, axis = 1)]
        return preds


def get_multi_pred(prot: str, glycans: list[str],  model: torch.nn.Module,  prot_dic: dict[str, list[float]], background_correction: bool = False,  correction_df: pd.DataFrame | None = None,
                   batch_size: int = 128,  libr: dict[str, int] | None = None,  flex: bool = False) -> list[float]:
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
  | libr (dict): dictionary of form glycoletter:index
  | flex (bool): depends on whether you use LectinOracle (False) or LectinOracle_flex (True); default:False\n
  | Returns:
  | :-
  | Returns dataframe of glycan sequences and predicted binding to prot"""
  if libr is None:
      libr = lib
  # Preparing dataset for PyTorch
  if flex:
      prot = prot_to_coded([prot])
      feature = prot * len(glycans)
  else:
      rep = prot_dic.get(prot, "new protein, no stored embedding")
      feature = [rep] * len(glycans)
  train_loader = dataset_to_dataloader(glycans, [0.99]*len(glycans),
                                         libr = libr, batch_size = batch_size,
                                         shuffle = False, extra_feature = feature)
  model = model.eval()
  res = []
  # Get predictions for each mini-batch
  for k in train_loader:
    x, y, edge_index, prot, batch = k.labels, k.y, k.edge_index, k.train_idx, k.batch
    x, y, edge_index, prot, batch = x.to(device), y.to(device), edge_index.to(device), prot.view(max(batch) + 1, -1).float().to(device), batch.to(device)
    pred = model(prot, x, edge_index, batch)
    res.extend(pred.detach().cpu().numpy())
  # Applying background correction of predictions
  if background_correction:
    correction_df = pd.Series(correction_df.pred.values,
                              index = correction_df.motif).to_dict()
    bg_res = [correction_df.get(j, 0) for j in glycans]
    if 0 in bg_res:
      print("Warning: not all glycans are in the correction_df; consider adding their background to correction_df")
    res = [a_i - b_i for a_i, b_i in zip(res, bg_res)]
  return res


def get_lectin_preds(prot: str, glycans: list[str],  model: torch.nn.Module,  prot_dic: dict[str, list[float]] | None = None, 
                     background_correction: bool = False, correction_df: pd.DataFrame | None = None,  batch_size: int = 128, 
                     libr: dict[str, int] | None = None,  sort: bool = True, flex: bool = False) -> pd.DataFrame:
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
  | libr (dict): dictionary of form glycoletter:index
  | sort (bool): whether to sort prediction results descendingly; default:True
  | flex (bool): depends on whether you use LectinOracle (False) or LectinOracle_flex (True); default:False\n
  | Returns:
  | :-
  | Returns dataframe of glycan sequences and predicted binding to prot"""
  if libr is None:
    libr = lib
  if correction_df is None:
    with resources.open_text("glycowork.ml", "glycowork_lectinoracle_background_correction.csv") as f:
        correction_df = pd.read_csv(f)
  if prot_dic is None and not flex:
    print("It seems you did not provide a dictionary of protein:ESM-1b representations. This is necessary.")
  preds = unwrap(get_multi_pred(prot, glycans, model, prot_dic,
                         batch_size = batch_size, libr = libr,
                         flex = flex))
  df_pred = pd.DataFrame({'motif': glycans, 'pred': preds})
  if background_correction:
    correction_dict = {motif: pred for motif, pred in zip(correction_df['motif'], correction_df['pred'])}
    for idx, row in df_pred.iterrows():
        motif = row['motif']
        if motif in correction_dict:
            df_pred.at[idx, 'pred'] -= correction_dict[motif]
  if sort:
    df_pred.sort_values('pred', ascending = True, inplace = True)
  return df_pred


def get_esm1b_representations(prots: list[str],  model: torch.nn.Module,  alphabet: Any) -> dict[str, list[float]]:
  """Retrieves ESM1b representations of protein for using them as input for LectinOracle\n
  | Arguments:
  | :-
  | prots (list): list of protein sequences (strings) that should be converted
  | model (ESM1b object): trained ESM1b model; from running esm.pretrained.esm1b_t33_650M_UR50S()
  | alphabet (ESM1b object): used for converting sequences; from running esm.pretrained.esm1b_t33_650M_UR50S()\n
  | Returns:
  | :-
  | Returns dictionary of the form protein sequence:ESM1b representation"""
  # model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
  batch_converter = alphabet.get_batch_converter()
  unique_prots = list(set(prots))
  data_list = [
        ('protein' + str(idx), prot[:min(len(prot), 1000)])
        for idx, prot in enumerate(unique_prots)
    ]
  _, _, batch_tokens = batch_converter(data_list)
  with torch.no_grad():
      results = model(batch_tokens, repr_layers = [33], return_contacts = False)
  token_representations = results["representations"][33]
  sequence_representations = [
        token_representations[i, 1:len(seq) + 1].mean(0)
        for i, (_, seq) in enumerate(data_list)
    ]
  prot_dic = {
        unique_prots[k]: s_rep.tolist()
        for k, s_rep in enumerate(sequence_representations)
    }
  return prot_dic


def get_Nsequon_preds(prots: list[str], model: torch.nn.Module, prot_dic: dict[str, list[float]]) -> pd.DataFrame:
  """Predicts whether an N-sequon will be glycosylated\n
  | Arguments:
  | :-
  | prots (list): list of protein sequences (strings), in the form of 20 AA + N + 20 AA; replace missing sequence with corr. number of 'z'
  | model (PyTorch object): trained NSequonPred-type model
  | prot_dic (dictionary): dictionary of type protein sequence:ESM1b representation\n
  | Returns:
  | :-
  | Returns dataframe of protein sequences and predicted likelihood of being an N-sequon"""
  reps = [prot_dic[k] for k in prots]
  # Preparing dataset for PyTorch
  dataset = SimpleDataset(reps, [0]*len(reps))
  loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = False)
  model = model.eval()
  preds = []
  # Get predictions for each mini-batch
  for x, _ in loader:
    x = x.to(device)
    pred = model(x)
    pred = [sigmoid(x) for x in pred.cpu().detach().numpy()]
    preds.extend(pred)
  df_pred = pd.DataFrame({
        'seq': prots,
        'glycosylated': preds
    })
  return df_pred
