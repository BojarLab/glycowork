import pandas as pd
import numpy as np
from importlib import resources
import math
import warnings
try:
    import torch
    # Choosing the right computing architecture
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
except ImportError:
    raise ImportError("<torch missing; did you do 'pip install glycowork[ml]'?>")
from glycowork.glycan_data.loader import lib, unwrap
from glycowork.motif.graph import compare_glycans
from glycowork.motif.tokenization import prot_to_coded
from glycowork.ml.processing import dataset_to_dataloader


class SimpleDataset:
    def __init__(self, x: list[float],  # input features
                 y: list[float]  # output labels
                 ) -> None:
        "Dataset class for Nsequon prediction"
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        inp = self.x[index]
        out = self.y[index]
        return torch.FloatTensor(inp), torch.FloatTensor([out])


def sigmoid(x: float  # input value
            ) -> float:  # sigmoid transformed value
    "Apply sigmoid transformation to input"
    if hasattr(x, 'item') or hasattr(x, 'dtype'):
        x = x.item()
    return 1 / (1 + math.exp(-x))


def glycans_to_emb(glycans: list[str],  # list of glycans in IUPAC-condensed
                   model: torch.nn.Module,  # trained graph neural network for analyzing glycans
                   libr: dict[str, int] | None = None,  # dictionary of form glycoletter:index
                   batch_size: int = 32,  # batch size used during training
                   rep: bool = True,  # True returns representations, False returns predicted labels
                   class_list: list[str] | None = None,  # list of unique classes to map predictions
                   multilabel = False  # whether to output predictions for a multilabel-task
                   ) -> pd.DataFrame | list[str]:  # dataframe of representations or list of predictions
    "Returns a dataframe of learned representations for a list of glycans"
    if libr is None:
        libr = lib
    # Preparing dataset for PyTorch
    glycan_loader = dataset_to_dataloader(glycans, range(len(glycans)), libr = libr, batch_size = batch_size,
                                          shuffle = False)
    res = []
    model = model.eval()
    # Get predictions for each mini-batch
    for data in glycan_loader:
        with torch.no_grad():
            x, y, edge_index, batch = data.labels, data.y, data.edge_index, data.batch
            x, y, edge_index, batch = x.to(device), y.to(device), edge_index.to(device), batch.to(device)
            pred, out = model(x, edge_index, batch, inference = True)
            # Unpacking and combining predictions
            res.extend(out.detach().cpu().numpy()) if rep else res.extend(pred.detach().cpu().numpy())
    return pd.DataFrame(res) if (rep or multilabel) else [class_list[k] for k in np.argmax(res, axis = 1)]


def get_multi_pred(prot: str,  # protein amino acid sequence
                   glycans: list[str],  # list of glycans in IUPAC-condensed
                   model: torch.nn.Module,  # trained LectinOracle-type model
                   prot_dic: dict[str, list[float]],  # dict of protein sequence:ESM1b representation
                   background_correction: bool = False,  # whether to correct predictions for background
                   correction_df: pd.DataFrame | None = None,  # background prediction for glycans
                   batch_size: int = 128,  # batch size used during training
                   libr: dict[str, int] | None = None,  # dict of glycoletter:index
                   flex: bool = False  # LectinOracle (False) or LectinOracle_flex (True)
                   ) -> list[float]:  # predicted binding values
    "Inner function to actually get predictions for lectin-glycan binding from LectinOracle-type model"
    if libr is None:
        libr = lib
    # Preparing dataset for PyTorch
    if flex:
        prot = prot_to_coded([prot])
        feature = prot * len(glycans)
    else:
        if prot not in prot_dic:
            raise KeyError(
                f"No stored embedding for the protein sequence of length {len(prot)} starting with '{prot[:20]}'; compute it with get_esmc_representations and add it to prot_dic, or use a LectinOracle_flex model with flex = True.")
        feature = [prot_dic[prot]] * len(glycans)
    train_loader = dataset_to_dataloader(glycans, [0.99] * len(glycans), libr = libr, batch_size = batch_size,
                                         label_type = torch.float, shuffle = False, extra_feature = feature)
    model = model.eval()
    res = []
    # Get predictions for each mini-batch
    for k in train_loader:
        with torch.no_grad():
            x, y, edge_index, prot, batch = k.labels, k.y, k.edge_index, k.train_idx, k.batch
            x, y, edge_index, prot, batch = x.to(device), y.to(device), edge_index.to(device), prot.view(int(batch.max()) + 1,
                                                                                                         -1).float().to(
                device), batch.to(device)
            pred = model(prot, x, edge_index, batch)
            res.extend(pred.detach().cpu().numpy())
    # Applying background correction of predictions
    if background_correction:
        correction_df = pd.Series(correction_df.pred.values, index = correction_df.motif).to_dict()
        # a background stored under a differently written but isomorphic sequence is still found, but compare_glycans rejects a different residue or branch count outright, so only those keys have to be scanned
        buckets = {}
        for k in correction_df:
            buckets.setdefault((k.count('('), k.count('[')), []).append(k)
        bg_res, missing = [], []
        for j in glycans:
            if j in correction_df:
                bg_res.append(correction_df[j])
            elif (hit := next((k for k in buckets.get((j.count('('), j.count('[')), ()) if compare_glycans(k, j)),
                              None)) is not None:
                bg_res.append(correction_df[hit])
            else:
                bg_res.append(0)
                missing.append(j)
        if missing:  # a background that happens to be exactly 0 is not a miss, so the glycans themselves have to be tracked
            warnings.warn(
                f"{len(missing)} of {len(glycans)} glycans are not in correction_df and stay uncorrected, which puts them on a different scale from the rest: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        res = [a_i - b_i for a_i, b_i in zip(res, bg_res)]
    return res


def get_lectin_preds(prot: str,  # protein amino acid sequence
                     glycans: list[str],  # list of glycans in IUPAC-condensed
                     model: torch.nn.Module,  # trained LectinOracle-type model
                     prot_dic: dict[str, list[float]] | None = None,  # dict of protein sequence:ESMC representation
                     background_correction: bool = False,  # whether to correct predictions for background
                     correction_df: pd.DataFrame | None = None,  # background prediction for glycans
                     batch_size: int = 128,  # batch size used during training
                     libr: dict[str, int] | None = None,  # dict of glycoletter:index
                     sort: bool = True,  # whether to sort prediction results descendingly
                     flex: bool = False  # LectinOracle (False) or LectinOracle_flex (True)
                     ) -> pd.DataFrame:  # glycan sequences and predicted binding
    "Wrapper that uses LectinOracle-type model for predicting binding of protein to glycans"
    if libr is None:
        libr = lib
    if correction_df is None and background_correction:
        with resources.files("glycowork.ml").joinpath("glycowork_lectinoracle_background_correction.csv").open(
                encoding = 'utf-8-sig') as f:
            correction_df = pd.read_csv(f)
    if prot_dic is None and not flex:
        raise ValueError(
            "It seems you did not provide a dictionary of protein:ESMC representations. This is necessary.")
    # Correcting inside get_multi_pred keeps one implementation, and that one matches the correction table by structure rather than by spelling
    preds = unwrap(get_multi_pred(prot, glycans, model, prot_dic, background_correction = background_correction,
                                  correction_df = correction_df, batch_size = batch_size, libr = libr, flex = flex))
    df_pred = pd.DataFrame({'motif': glycans, 'pred': preds})
    if sort:
        df_pred.sort_values('pred', ascending = False, inplace = True)
    return df_pred


def _clean_protein_sequences(prots: list[str],  # protein sequences to filter
                             max_len: int = 1000  # maximum sequence length to keep
                             ) -> list[str]:
    """Filter protein sequences to valid ESM input and remove duplicates."""
    valid_protein_char = set("ACDEFGHIKLMNPQRSTVWYBXZOU.-")
    cleaned = []
    skipped = 0
    for raw_seq in prots:
        if not isinstance(raw_seq, str):
            skipped += 1
            continue
        seq = raw_seq.strip().upper()
        if not seq:
            skipped += 1
            continue
        if set(seq) - valid_protein_char:
            skipped += 1
            continue
        cleaned.append(seq[:max_len])
    if skipped:
        warnings.warn(
            f"Skipped {skipped} invalid or empty protein sequence(s) before ESM encoding.",
            stacklevel = 2,
        )
    unique = list(dict.fromkeys(cleaned))
    if not unique:
        raise ValueError("No valid protein sequences remained after cleaning.")
    return unique


def get_esmc_representations(prots: list[str],  # list of protein sequences to convert
                             model: torch.nn.Module,  # trained ESMC model
                             ) -> dict[str, list[float]]:  # dict of protein sequence:ESMC-300M representation
    "Retrieves ESMC-300M representations of protein for using them as input for LectinOracle"
    # from esm.models.esmc import ESMC
    # model = ESMC.from_pretrained("esmc_300m").to(device)
    prots = _clean_protein_sequences(prots)
    try:
        from esm.sdk.api import ESMProtein, LogitsConfig
        use_esm_api = True
    except ImportError:
        use_esm_api = False
        # Only raise the error if we're not in a testing context
        if not hasattr(model, 'encode') or not hasattr(model, 'logits'):
            raise ImportError(
                "<To use this function, you will need to install esm (pip install esm), which is not a dependency of glycowork>")

    def prot_to_ESMC(seq: str):
        if use_esm_api:
            protein_tensor = model.encode(ESMProtein(sequence = seq))
            logits_output = model.logits(protein_tensor, LogitsConfig(sequence = True, return_embeddings = True))
        else:
            # For test scenarios with mock model
            protein_tensor = model.encode(seq)
            logits_output = model.logits(protein_tensor, None)
        return torch.mean(logits_output.embeddings, dim = 1).squeeze().tolist()

    return {p: prot_to_ESMC(p) for p in prots}


def get_Nsequon_preds(prots: list[str],  # 20 AA + N + 20 AA sequences; replace missing with 'z'
                      model: torch.nn.Module,  # trained NSequonPred-type model
                      prot_dic: dict[str, list[float]]  # dict of protein sequence:ESM1b representation
                      ) -> pd.DataFrame:  # protein sequences and predicted likelihood
    "Predicts whether an N-sequon will be glycosylated"
    reps = [prot_dic[k] for k in prots]
    # Preparing dataset for PyTorch
    dataset = SimpleDataset(reps, [0] * len(reps))
    loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = False)
    model = model.eval()
    preds = []
    # Get predictions for each mini-batch
    with torch.no_grad():
        for x, _ in loader:
            pred = model(x.to(device))
            preds.extend((1 / (1 + np.exp(-pred.cpu().numpy().astype(float)))).ravel().tolist())
    df_pred = pd.DataFrame({
        'seq': prots,
        'glycosylated': preds
    })
    return df_pred
