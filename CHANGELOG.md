# Changelog

## [1.9.1]

### glycan_data
#### loader
##### Changed 🔄
- Changed the `GlycoDataFrame` attribute `_name` to `_glyco_name` to avoid shadowing the `pandas` attribute

### motif
#### tokenization
##### Added ✨

##### Changed 🔄

##### Fixed 🐛
- Fixed `glycan_to_composition` handling of narrow monosaccharide modification wildcards such as `HexNAc4/6S` (b3ad039)

##### Deprecated ⚠️

#### analysis
##### Added ✨
- `get_biodiversity` now has a new `circadian` API that supports analyzing whether changes in alpha-diversity (internally using `get_jtk`) and beta-diversity, using a distance-based redundancy analysis (db-RDA) regressing the Gower-centered dissimilarity matrix onto a cosinor design, are rhythmic (b4b29c7)
- `select_grouping` now supports DAG-based grouping via `get_motif_dag` (a918a2e)
- `get_differential_expression`, `get_glycanova`, `get_time_series`, and `get_jtk` now can correct DAG-based groups by grouped Benjamini-Hochberg in the case of motif-based analysis if `grouped_BH=True` (a918a2e)
- Added the new `grouped_BH` keyword argument to `get_glycanova`, `get_time_series`, and `get_jtk` (a918a2e)
- `get_differential_expression` and `get_glycanova` now also announce which children motifs support the change in a parent motif (and whether a portion of the change cannot be explained by children motifs) (a918a2e)
- `get_differential_expression` and `get_glycanova` now also report a "Redistribution p-val", which tests whether the motif is now found in *different* sequence contexts (e.g., `Internal_LacNAc_type2` might not change in abundance but might be redistributed from sialyl-LacNAc to H-type motifs) (a918a2e)
- `get_roc` now also outputs the selected features if `multi_score=True` (5483f3c)
- functions in `.analysis` now can read the stored metadata automatically and act accordingly, supporting easy-calls like `get_differential_expression(glycomics_data_loader.human_brain_N_PMID38343116)` (5483f3c)

##### Changed 🔄
- `get_pca` now correctly filters out redundant motifs for the PCA analysis (a918a2e)
- `get_differential_expression`, `get_glycanova`, `get_time_series`, and `get_jtk` now default to `grouped_BH=True` if they are run in motif-analysis mode (`motifs=True`) (7849b15)

##### Fixed 🐛
- `get_time_series` and `get_jtk` now correctly do motif quantification followed by CLR/ALR (instead of the other way around) in case of motif-analysis (a918a2e)

#### processing
##### Changed 🔄
- `de_wildcard_glycoletter` now also supports narrow monosaccharide wildcards like `Gal/Glc` (5483f3c)

##### Fixed 🐛
- Fixed `canonicalize_iupac` messing up narrow modification wildcards (e.g., `Gal3/6S`) in side branches (3a02eff)
- Fixed `canonicalize_iupac` sometimes ordering multiple floating bits wrongly (b6689a5)

#### draw
##### Changed 🔄
- Drawing "forbidden" monosaccharides, such as in `GlycoDraw("Gal(b1-3)[!GlcNAc(b1-6)]GalNAc")`, now automatically makes the forbidden monosaccharides and their linkages transparent (3621350)

#### graph
##### Changed 🔄
- `subgraph_isomorphism` and `compare_glycans` have been made more performant (5483f3c)

##### Fixed 🐛
- Fixed `subgraph_isomorphism` counting too many isomorphic matches in certain scenarios of linkage ambiguity (81244c9)
- `get_linkage_number` now always handles narrow linkage wildcards correctly for branch sorting purposes (d222f41)

#### annotate
##### Added ✨
- Added `get_motif_dag` to build a containment DAG of connected motifs, for comparative glycomics analyses in `analysis` (a918a2e)

##### Changed 🔄
- Refined counting of `Terminal_` motifs in `annotate_dataset` (a918a2e)
- `deduplicate_motifs` will now (given the choice) always prefer the specified motif over the unspecified motif, all else being equal (e.g., `Fuc` > `dHex`) (81e769c)
- `annotate_dataset` will no longer split signal between pairs such as `Gal(b1-4)GlcNAc` and `Gal(b1-4)GlcNAc-ol` in free oligosaccharides (3737633)

### network
#### biosynthesis
##### Changed 🔄
- In `construct_network`, `edge_type=enzyme` will now assign glycan class-specific enzymes, if possible (e.g., only ST3GAL4 for N-glycans instead of all ST3GALs) (cb262b3)

### glycan_data
#### stats
##### Added ✨
- `clr_transformation` now has a new `reference` keyword argument, to optionally specify from which variables the geometric mean should be constructed (a918a2e)
- `MissForest` now runs left-censored draws where missingness is intensity-dependent (MNAR) and Random Forest for the rest (MAR), where intensity-dependence is estimated via logistic regression (7849b15)
- `get_alphaN` now has a new `verbose` keyword argument to suppress its print (which is now default `False`, since its output will now be stored as dataframe attributes) (5483f3c)

##### Changed 🔄
- `hotellings_t2` is now more robust to tiny groups (a918a2e)
- `TST_grouped_benjamini_hochberg` is now more robust to groups with π0 = 1 (c902297)
- `replace_outliers_winsorization` now takes in the entire `df` as input, instead of only one `row` (5483f3c)
- `replace_outliers_winsorization` and `omega_squared` are now much faster (5483f3c)

##### Fixed 🐛
- Fixed sum of squares calculation in `calculate_permanova_stat` (5483f3c)

### ml
#### model_training
##### Fixed 🐛
- Fixed double softmax in `train_model` (5483f3c)