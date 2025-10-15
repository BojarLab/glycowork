from collections import defaultdict
import copy
import os
import pickle
from typing import Literal, Optional, Union
from pathlib import Path
import urllib.request

import platformdirs
import glyles
import numpy as np
import pandas as pd
import networkx as nx
from yaml import warnings
try:
    from datasail.sail import datasail
    from nxontology.imports import from_file
    import numpy_indexed as npi
    import pyarrow
    from pyarrow import csv
except ImportError:
    raise ImportError("<One or multiple packages of [nxontology, datasail, numpy_indexed, pyarrow] are missing; did you do 'pip install glycowork[gym]'?>")

from glycowork.glycan_data.loader import build_custom_df, df_glycan, df_species, glycan_binding
from glycowork.motif.processing import canonicalize_iupac


# General constants
MIN_FREQUENCY = 15
# Creating and defining a glycowork cache directory
CACHE_DIR = Path(os.getenv("GLYCOWORK_CACHE", platformdirs.user_cache_dir("glycowork")))
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Constants for tissue resolution
UBERON_ROOTS = [
    "UBERON:0000045", "UBERON:0000476", "UBERON:0003102", "UBERON:0004119",
    "UBERON:0004120", "UBERON:0004121", "UBERON:0005090", "UBERON:0005162",
    "UBERON:0005389", "UBERON:0009856", "UBERON:0010000", "UBERON:0010314",
    "UBERON:0015212", "UBERON:0034768", "UBERON:0000173", "UBERON:0000174",
    "UBERON:0000456", "UBERON:0001011", "UBERON:0001968", "UBERON:0006314",
    "UBERON:0006535", "UBERON:0008944", "UBERON:0022293", "UBERON:0002050",
    "CL:0000039", "CL:0000151", "CL:0000211", "CL:0000219",
    "CL:0000225", "CL:0000325", "CL:0000329", "CL:0000413",
    "CL:0000988", "CL:0002242", "CL:0011115"
]
UBERON_ROOT_SET = set(UBERON_ROOTS)
UBERON_ROOT_MAP = {root: r for r, root in enumerate(UBERON_ROOTS)}

# Constants for spectrum binning
MIN_MZ = 39.741
MAX_MZ = 3000
NUM_BINS = 2048
BIN_SIZE = (MAX_MZ - MIN_MZ) / NUM_BINS
vector = np.arange(MIN_MZ, MAX_MZ, BIN_SIZE)


class SMILESStorage:
    def __init__(self, path: Path | str | None = None):
        """
        Initialize the wrapper around a dict.

        Args:
            path: Path to the directory. If there's a glycan_storage.pkl, it will be used to fill this object,
                otherwise, such file will be created.
        """
        self.path = Path(path or "data") / "smiles_storage.pkl"

        # Fill the storage from the file
        self.data = self._load()

    def close(self) -> None:
        """
        Close the storage by storing the dictionary at the location provided at initialization.
        """
        with open(self.path, "wb") as out:
            pickle.dump(self.data, out)

    def query(self, iupac: str) -> Optional[str]:
        """
        Query the storage for a IUPAC string.

        Args:
            iupac: The IUPAC string of the query glycan

        Returns:
            A HeteroData object corresponding to the IUPAC string or None, if the IUPAC string could not be processed.
        """
        if iupac not in self.data:
            if "{" in iupac or "?" in iupac:
                self.data[iupac] = None
                return None
            try:
                translation = glyles.convert(iupac)
                self.data[iupac] = translation[0][1] if isinstance(translation, list) else translation
            except Exception:
                self.data[iupac] = None
        return copy.deepcopy(self.data[iupac])

    def _load(self) -> dict[str, Optional[str]]:
        """
        Load the internal dictionary from the file, if it exists, otherwise, return an empty dict.

        Returns:
            The loaded (if possible) or an empty dictionary
        """
        if self.path.exists():
            with open(self.path, "rb") as f:
                return pickle.load(f)
        return {}


def standardize_iupac(iupac: str) -> Optional[str]:
    """
    Standardize the IUPAC name of a glycan.

    Args:
        iupac: The IUPAC name to standardize.

    Returns:
        The standardized IUPAC name, or None if it could not be standardized.
    """
    try:
        return canonicalize_iupac(iupac)
    except:
        return None


def to_oh(names: list[str]) -> np.ndarray:
    """
    Convert a list of UBERON IDs to a one-hot encoding vector.

    Args:
        names: List of UBERON IDs.

    Returns:
        A one-hot encoding vector corresponding to the UBERON IDs.
    """
    vec = np.zeros(len(UBERON_ROOTS))
    for name in names:
        vec[UBERON_ROOT_MAP[name]] = 1
    return vec


def bin_intensities(peak_d, frames):
    """
    Sums up intensities for each bin across a spectrum.

    Args:
        peak_d (dict): dictionary of form (fragment) m/z : intensity
        frames (list): m/z boundaries separating each bin

    Returns:
        a list of binned intensities
    """
    binned_intensities  = np.zeros(len(frames))
    mzs = np.array(list(peak_d.keys()), dtype = 'float32')
    intensities = np.array(list(peak_d.values()))
    bin_indices = np.digitize(mzs, frames, right = True)
    unique_bins, summed_intensities = npi.group_by(bin_indices).sum(intensities)
    binned_intensities[unique_bins - 1] = summed_intensities
    binned_intensities = binned_intensities / np.sum(binned_intensities)
    return list(binned_intensities)


def iupac_mask(tab: pyarrow.Table, iupac: str) -> np.ndarray:
    """
    Create a mask that removes all but 1000 samples of a specific IUPAC string.

    Args:
        tab: The pyarrow table to create the mask for.
        iupac: The IUPAC string to create the mask for.
    
    Returns:
        A boolean numpy array that can be used as a mask for the table.
    """
    to_sample = []
    for i, array in enumerate(tab["IUPAC"]):
        if array.as_py() == iupac:
            to_sample.append(i)
    mask = [True] * len(tab)
    to_sample = np.array(to_sample)
    np.random.shuffle(to_sample)
    for i in to_sample[:-1000]:
        mask[i] = False
    return np.array(mask)


def build_glycosylation(top_k: int = -1) -> tuple[pd.DataFrame, dict]:
    """
    Download glycosylation data, process it, and save it as a tsv file.

    Args:
        top_k: If > 0, only use the top_k entries from the original dataset.

    Returns:
        The processed glycosylation data as a DataFrame and a mapping from class names to integer labels.
    """
    # Read in the glycosylation data
    link = df_glycan[["glycan", "glycan_type"]]
    if top_k > 0:
        link = link.sample(frac=1).head(top_k)

    # Standardize the IUPAC names
    link["IUPAC"] = link["glycan"].apply(standardize_iupac)
    link = link[link["IUPAC"].notna()]

    # Drop duplicate IUPAC names
    link.drop_duplicates("IUPAC", inplace=True)

    # Translate IUPAC names to SMILES and remove those that cannot be converted
    smiles = SMILESStorage(CACHE_DIR)
    link["SMILES"] = link["IUPAC"].apply(lambda x: smiles.query(x))
    smiles.close()
    link = link[link["SMILES"] != ""]
    link = link[link["SMILES"].notna()]
    # Remove classes with less than 15 annotations
    keep = set([k for k, v in dict(link["glycan_type"].value_counts()).items() if v >= MIN_FREQUENCY])
    link = link[link["glycan_type"].isin(keep)]

    # Compute datasplit with datasail
    e_splits, _, _ = datasail(
        techniques=["I1e"],
        splits=[7, 2, 1],
        names=["train", "val", "test"],
        e_type="O",
        e_data={x: x for x in link["IUPAC"].tolist()},
        e_strat=dict(link[["IUPAC", "glycan_type"]].values),
        epsilon=0.2,
        delta=0.2,
    )
    link["split"] = link["IUPAC"].map(lambda x: e_splits["I1e"][0].get(x, None))

    class_map = {label: i for i, label in enumerate(link["glycan_type"].unique())}
    link["label"] = link["glycan_type"].map(class_map)
    link = link[["IUPAC", "SMILES", "label", "split"]]

    return link, class_map


def build_taxonomy(
        level: Literal["Domain", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"],
        top_k: int = -1,
) -> pd.DataFrame:
    """
    Extract taxonomy data at a specific level, process it, and save it as a tsv file.

    Args:
        root: The root directory to save the data to.
        level: The taxonomic level to extract the data from.
        top_k: If > 0, only use the top_k entries from the original dataset.

    Returns:
        The processed taxonomy data as a DataFrame.
    """
    # Read in the taxonomy data
    tax = df_species[["glycan", level]]
    if top_k > 0:
        tax = tax.sample(frac=1).head(top_k)

    # Standardize the IUPAC names
    tax["IUPAC"] = tax["glycan"].apply(standardize_iupac)
    tax = tax[tax["IUPAC"].notna()]

    # Remove all glycans that do not have a valid taxonomic level
    tax = tax[tax[level] != "undetermined"]
    
    # One-hot encode the individual classes and collate them for glycans that are the same
    tax = pd.concat([tax["IUPAC"], pd.get_dummies(tax[level])], axis=1)
    tax = tax.groupby('IUPAC').agg("sum").reset_index()

    # Ensure onehot values for classes are 0 or 1
    classes = [x for x in tax.columns if x != "IUPAC"]
    tax[classes] = tax[classes].map(lambda x: min(1, x))

    # Convert IUPAC names to SMILES and remove those that cannot be converted
    smiles = SMILESStorage(CACHE_DIR)
    tax["SMILES"] = tax["IUPAC"].apply(lambda x: smiles.query(x) or "")
    smiles.close()
    tax = tax[tax["SMILES"] != ""]
    tax = tax[tax["SMILES"].notna()]

    # Remove all classes that have less than 10 annotations and glycans with no annotations
    cols = [x for x in tax.columns if x not in {"IUPAC", "SMILES"} and tax[x].values.sum() >= MIN_FREQUENCY]
    tax = tax[["IUPAC", "SMILES"] + cols]
    tax = tax[tax[cols].values.sum(axis=1) > 0.5].astype({c: int for c in cols})

    # Extract the stratification for the datasail splits
    strat = {}
    for _, row in tax.iterrows():
        strat[row["IUPAC"]] = list(row[cols].values)

    # Compute the datasail splits
    e_splits, _, _ = datasail(
        techniques=["I1e"],
        splits=[7, 2, 1],
        names=["train", "val", "test"],
        e_type="O",
        e_data={x: x for x in tax["IUPAC"].tolist()},
        e_strat=strat,
        epsilon=0.2,
        delta=0.2,
    )
    tax["split"] = tax["IUPAC"].map(lambda x: e_splits["I1e"][0].get(x, None))
    
    return tax


def build_tissue(top_k: int = -1) -> pd.DataFrame:
    """
    Load the tissue data, process it, and save it as a tsv file.

    Args:
        top_k: If > 0, only use the top_k entries from the original dataset.
    
    Returns:
        The processed tissue data as a DataFrame.
    """
    # Read in the UBERON ontology and the glycan tissue data
    # UBERON = from_file("uberon.owl").graph
    urllib.request.urlretrieve("http://purl.obolibrary.org/obo/uberon.owl", CACHE_DIR / "uberon.owl")
    UBERON = from_file(CACHE_DIR / "uberon.owl").graph
    df = build_custom_df(df_glycan, "df_tissue")[["glycan", "tissue_id"]]
    if top_k > 0:
        df = df.sample(frac=1).head(top_k)

    # Remove all unknown tissue ids
    df = df[df["tissue_id"].apply(lambda x: x in UBERON.nodes)]
    
    # Canonicalize the IUPAC names
    df["IUPAC"] = df["glycan"].apply(standardize_iupac)
    df = df[df["IUPAC"].notna()]

    # Bubble up tissue ids to our roots of the UBERON ontology
    df["uberon"] = df["tissue_id"].apply(lambda x: list(set(nx.ancestors(UBERON, x)).intersection(UBERON_ROOTS)))

    # Compute onehot encodings for the UBERON IDs
    vecs = np.stack([to_oh(names) for names in df["uberon"].values])
    tissue = pd.concat([df[["IUPAC"]].reset_index(), pd.DataFrame(vecs, columns=UBERON_ROOTS, dtype=int)], axis=1)
    
    # Group by IUPAC and sum the one-hot encodings
    tissue = tissue.groupby("IUPAC").agg("sum").reset_index()

    # Convert IUPAC names to SMILES and remove those that cannot be converted
    smiles = SMILESStorage(CACHE_DIR)
    tissue["SMILES"] = tissue["IUPAC"].apply(lambda x: smiles.query(x))
    smiles.close()
    tissue = tissue[tissue["SMILES"] != ""]
    tissue = tissue[tissue["SMILES"].notna()]

    # Remove all classes that have less than 15 annotations
    keep = ["IUPAC", "SMILES"] + [x for x in tissue.columns if ":" in x and tissue[x].values.sum() >= MIN_FREQUENCY]
    tissue = tissue[keep]
    
    # Remove all glycans that do not have any tissue annotations
    class_names = [c for c in tissue.columns if ":" in c]
    tissue = tissue[tissue[class_names].values.sum(axis=1) > 0.5].astype({c: int for c in class_names})
    tissue[class_names] = tissue[class_names].applymap(lambda x: min(x, 1))
    
    # Extract the tissue stratification for the datasail splits
    strat = {}
    for _, row in tissue.iterrows():
        strat[row["IUPAC"]] = row[class_names].values.tolist()
    
    # Convert the IUPAC names to SMILES for the datasail splits
    e_splits, _, _ = datasail(
        techniques=["I1e"],
        splits=[7, 2, 1],
        names=["train", "val", "test"],
        e_type="O",
        e_data={x: x for x in tissue["IUPAC"].tolist()},
        e_strat=strat,
        epsilon=0.3,
        delta=0.3,
    )
    tissue["split"] = tissue["IUPAC"].map(lambda x: e_splits["I1e"][0].get(x, None))
    return tissue


def build_lgi(top_k: int = -1) -> Union[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]]:
    """
    Process the glycan-binding data into the GlycoGym format.

    Args:
        top_k: If > 0, only use the top_k entries from the original dataset. If -1, return both the full dataset and a 150k subset.
    
    Returns:
        Three dataframes (or tuples of dataframes) corresponding to the random, lectin, and glycan splits.
    """
    # Read in the glycan-binding data
    glycan_binding.index = glycan_binding["target"]
    glycan_binding.drop(columns=["target", "protein"], inplace=True)
    stack = glycan_binding.stack()

    glycans = {f"Gly{i:04d}": iupac for i, iupac in enumerate(glycan_binding.columns)}
    glycans.update({iupac: f"Gly{i:04d}" for i, iupac in enumerate(glycan_binding.columns)})

    lectins = {f"Lec{i:04d}": aa_seq for i, aa_seq in enumerate(glycan_binding.index)}
    lectins.update({aa_seq: f"Lec{i:04d}" for i, aa_seq in enumerate(glycan_binding.index)})

    df = pd.DataFrame([(lectins[aa_seq], glycans[iupac], binding) for (aa_seq, iupac), binding in stack.items()], columns=["lectin", "glycan", "binding"])
    if top_k > 0:
        df = df.sample(frac=1).head(top_k)

    smiles = SMILESStorage(CACHE_DIR)
    valid_glycans = list(filter(lambda x: smiles.query(glycans[x]) or "" != "", df["glycan"].unique()))
    smiles.close()

    valid_glycans = set(valid_glycans)
    df = df[df["glycan"].isin(valid_glycans)]

    _, _, inter_split = datasail(
        techniques=["R", "I1e", "I1f"],
        splits=[7, 2, 1],
        names=["train", "val", "test"],
        inter=[(x, y) for x, y in df[["lectin", "glycan"]].values],
        e_type="P",
        e_data={idx: lectins[idx] for idx in df["lectin"].unique()},
        e_weights=df["lectin"].value_counts().to_dict(),
        f_type="O",
        f_data={idx: glycans[idx] for idx in df["glycan"].unique()},
        f_weights=df["glycan"].value_counts().to_dict(),
    )

    df.rename(columns={"binding": "y"}, inplace=True)

    df_r = df.copy()
    df_r["split"] = df[["lectin", "glycan"]].apply(lambda x: inter_split["R"][0][x[0], x[1]], axis=1)
    df_r = df_r[["lectin", "glycan", "y", "split"]]
    if top_k == -1:
        df_r_150k = df_r.sample(150000)
    
    df_cl = df.copy()
    df_cl["split"] = df_cl[["lectin", "glycan"]].apply(lambda x: inter_split["I1e"][0][x[0], x[1]], axis=1)
    df_cl = df_cl[["lectin", "glycan", "y", "split"]]
    if top_k == -1:
        df_cl_150k = df_cl.sample(150000)

    df_cg = df.copy()
    df_cg["split"] = df_cg[["lectin", "glycan"]].apply(lambda x: inter_split["I1f"][0][x[0], x[1]], axis=1)
    df_cg = df_cg[["lectin", "glycan", "y", "split"]]
    if top_k == -1:
        df_cg_150k = df_cg.sample(150000)

    if top_k == -1:
        return (df_r, df_r_150k), (df_cl, df_cl_150k), (df_cg, df_cg_150k)
    return df_r, df_cl, df_cg


def build_spectrum(root) -> pd.DataFrame:
    """
    Process the glycan spectrum data into the GlycoGym format.

    Args:
        root: The root directory where the spectrum data is stored.
    
    Returns:
        A dataframe corresponding to the spectrum data.
    """

    with open(root / "reduced_neg_not_Olinked.pkl", "rb") as f:
        olinked = pickle.load(f)
    print("O-Linked spectra:", len(olinked))
    with open(root / "FragmentFactory_dataset.pkl", "rb") as f:
        ff = pickle.load(f)
    print("Factory spectra :", len(ff))

    dataset = []
    for d, data in enumerate([olinked, ff]):
        for i, (_, entry) in enumerate(data.iterrows()):
            print(f"\r{d + 1}/2: {i + 1}/{len(data)}", end="")
            red_mz = entry["reducing_mass"]
            peaks = bin_intensities(entry["peak_d"], vector)
            dataset.append({"IUPAC": entry["glycan"], "red_mz": red_mz, **{f"bin_{i}": p for i, p in enumerate(peaks)}})
            
    table = pyarrow.Table.from_pylist(dataset)

    iupac_counts = defaultdict(int)
    for array in table["IUPAC"]:
        iupac_counts[array.as_py()] += 1

    smiles = SMILESStorage(CACHE_DIR)
    smiles_map = {iupac: smiles.query(iupac) or "" for iupac in iupac_counts.keys()}
    smiles.close()

    rare_iupacs = set(k for k, v in iupac_counts.items() if v < 15)
    freq_iupacs = set(k for k, v in iupac_counts.items() if v > 1000)

    mask = np.array([iupac.as_py() not in rare_iupacs and len(smiles_map[iupac.as_py()]) > 10 for iupac in table["IUPAC"]])
    for iupac in freq_iupacs:
        mask &= iupac_mask(table, iupac)

    table = table.filter(mask)
    iupacs = set(x.as_py() for x in table["IUPAC"])

    e_splits, _, _ = datasail(
        techniques=["I1e"],
        names=["train", "val", "test"],
        splits=[7, 2, 1],
        e_type="O",
        e_data={i: i for i in iupacs},
        e_weights={str(i): min(iupac_counts[i], 1000) for i in iupacs},
    )
    table = table.append_column("split", pyarrow.array([e_splits["I1e"][0][iupac.as_py()] for iupac in table["IUPAC"]]))

    return table


def build_glycogym(spectrum_root: Union[str, Path], output: Optional[Union[str, Path]] = None) -> None:
    """
    Build all datasets used in GlycoGym.

    Args:
        root: The root directory to save the data to. If None, use the current working directory.
        top_k: If > 0, only use the top_k entries from the original dataset. If -1, return both the full dataset and a 150k subset for the LGI data.
    
    Returns:
        A dictionary with the different datasets as dataframes.
    """
    if output is None:
        warnings.warn("No output directory provided, using current working directory. This process will write ~1.5GB of disk space.")
    root = Path(output or ".")
    root.mkdir(exist_ok=True, parents=True)

    print("Building glycosylation data...")
    glyco, glyco_map = build_glycosylation()
    glyco.to_csv(root / "glycosylation.tsv", sep="\t", index=False)
    with open(root / "glycosylation_map.pkl", "w") as f:
        for k, v in glyco_map.items():
            print(f"{k}\t{v}", file=f)

    print("Building taxonomic data...")
    for level in ["Domain", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]:
        print(f" - {level}")
        tax = build_taxonomy(level)
        tax.to_csv(root / f"taxonomy_{level.lower()}.tsv", sep="\t", index=False)

    print("Building tissue data...")
    tissue = build_tissue()
    tissue.to_csv(root / "tissue.tsv", sep="\t", index=False)

    print("Building LGI data...")
    df_r, df_cl, df_cg = build_lgi()
    df_r[0].to_csv(root / "lgi_random.tsv", sep="\t", index=False)
    df_r[1].to_csv(root / "lgi_random_150k.tsv", sep="\t", index=False)
    df_cl[0].to_csv(root / "lgi_cold_lectin.tsv", sep="\t", index=False)
    df_cl[1].to_csv(root / "lgi_cold_lectin_150k.tsv", sep="\t", index=False)
    df_cg[0].to_csv(root / "lgi_cold_glycan.tsv", sep="\t", index=False)
    df_cg[1].to_csv(root / "lgi_cold_glycan_150k.tsv", sep="\t", index=False)
    
    print("Building spectrum data...")
    spectrum = build_spectrum(spectrum_root)
    csv.write_csv(spectrum, root / "spectrum.tsv", write_options=csv.WriteOptions(delimiter="\t", include_header=True))
    
    print("The structural datasets need to be build directly from GlyConnect.")
