import pandas as pd
import numpy as np
from importlib import resources
from typing import Any, Dict, List, Optional, Tuple, Union
import math
try:
    import torch
    # Choosing the right computing architecture
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
except ImportError:
  raise ImportError("<torch missing; did you do 'pip install glycowork[ml]'?>")
from glycowork.glycan_data.loader import lib, unwrap
from glycowork.motif.tokenization import prot_to_coded
from glycowork.ml.processing import dataset_to_dataloader


class SimpleDataset:
  def __init__(self, x: List[float], # input features
                 y: List[float] # output labels
                ) -> None:
    "Dataset class for Nsequon prediction"
    self.x = x
    self.y = y

  def __len__(self) -> int:
    return len(self.x)

  def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    inp = self.x[index]
    out = self.y[index]
    return torch.FloatTensor(inp), torch.FloatTensor([out])


def sigmoid(x: float # input value
          ) -> float: # sigmoid transformed value
  "Apply sigmoid transformation to input"
  if hasattr(x, 'item') or hasattr(x, 'dtype'):
      x = x.item()
  return 1 / (1 + math.exp(-x))


def glycans_to_emb(glycans: List[str], # list of glycans in IUPAC-condensed
                  model: torch.nn.Module, # trained graph neural network for analyzing glycans
                  libr: Optional[Dict[str, int]] = None, # dictionary of form glycoletter:index
                  batch_size: int = 32, # batch size used during training
                  rep: bool = True, # True returns representations, False returns predicted labels
                  class_list: Optional[List[str]] = None # list of unique classes to map predictions
                 ) -> Union[pd.DataFrame, List[str]]: # dataframe of representations or list of predictions
    "Returns a dataframe of learned representations for a list of glycans"
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
        res.extend(out.detach().cpu().numpy()) if rep else res.extend(pred.detach().cpu().numpy())
    return pd.DataFrame(res) if rep else [class_list[k] for k in np.argmax(res, axis = 1)]


def get_multi_pred(prot: str, # protein amino acid sequence
                  glycans: List[str], # list of glycans in IUPAC-condensed
                  model: torch.nn.Module, # trained LectinOracle-type model
                  prot_dic: Dict[str, List[float]], # dict of protein sequence:ESM1b representation
                  background_correction: bool = False, # whether to correct predictions for background
                  correction_df: Optional[pd.DataFrame] = None, # background prediction for glycans
                  batch_size: int = 128, # batch size used during training
                  libr: Optional[Dict[str, int]] = None, # dict of glycoletter:index
                  flex: bool = False # LectinOracle (False) or LectinOracle_flex (True)
                 ) -> List[float]: # predicted binding values
  "Inner function to actually get predictions for lectin-glycan binding from LectinOracle-type model"
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
                                         libr = libr, batch_size = batch_size, label_type = torch.float,
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


def get_lectin_preds(prot: str, # protein amino acid sequence
                    glycans: List[str], # list of glycans in IUPAC-condensed
                    model: torch.nn.Module, # trained LectinOracle-type model
                    prot_dic: Optional[Dict[str, List[float]]] = None, # dict of protein sequence:ESM1b representation
                    background_correction: bool = False, # whether to correct predictions for background
                    correction_df: Optional[pd.DataFrame] = None, # background prediction for glycans
                    batch_size: int = 128, # batch size used during training
                    libr: Optional[Dict[str, int]] = None, # dict of glycoletter:index
                    sort: bool = True, # whether to sort prediction results descendingly
                    flex: bool = False # LectinOracle (False) or LectinOracle_flex (True)
                   ) -> pd.DataFrame: # glycan sequences and predicted binding
  "Wrapper that uses LectinOracle-type model for predicting binding of protein to glycans"
  if libr is None:
    libr = lib
  if correction_df is None:
    with resources.files("glycowork.ml").joinpath("glycowork_lectinoracle_background_correction.csv").open(encoding = 'utf-8-sig') as f:
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


def get_esm1b_representations(prots: List[str], # list of protein sequences to convert
                            model: torch.nn.Module, # trained ESM1b model
                            alphabet: Any # used for converting sequences
                           ) -> Dict[str, List[float]]: # dict of protein sequence:ESM1b representation
  "Retrieves ESM1b representations of protein for using them as input for LectinOracle"
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


def get_Nsequon_preds(prots: List[str], # 20 AA + N + 20 AA sequences; replace missing with 'z'
                     model: torch.nn.Module, # trained NSequonPred-type model
                     prot_dic: Dict[str, List[float]] # dict of protein sequence:ESM1b representation
                    ) -> pd.DataFrame: # protein sequences and predicted likelihood
  "Predicts whether an N-sequon will be glycosylated"
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
