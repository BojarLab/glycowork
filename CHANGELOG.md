# Changelog

## [1.10.2]

### motif
#### smiles
##### Added ✨
- Added support for phospho-/sulfodiester in SMILES conversion (e5e8631)
- Added support for acyl chain modifications (01253cb)

##### Changed 🔄

##### Fixed 🐛
- Fixed `(2R,3R) L-threo` to `D-erythro (2S,3R)` sphingosine as a default for ceramide (01253cb)
- Fixed amino acid configuration sometimes not defaulting to `L` (01253cb)
- Fixed handling of `1N`-modified monosaccharides such as `GlcNAc1N` and halogen-substituted monosaccharides such as `Gal6F` (01253cb)
- Fixed alditols engaging in glycosidic linkages (01253cb)
- Various other fixes for more exotic modifications (01253cb)

##### Deprecated ⚠️

#### draw
##### Added ✨
- Added the new `plot_glycans_grid` function to draw glycans with a shared symbol size into a grid (0967d8a)
- `GlycoDraw` now also accepts alternative composition formats (e.g., `{'Hex': 5}`) (0967d8a)
- `GlycoDraw` now accepts lists as input, which are fed into the new `plot_glycans_grid` (0967d8a)
- `plot_glycans_excel` can now pass on styling keyword arguments (e.g., `compact`, `vertical`, etc) to `GlycoDraw` (0967d8a)

##### Changed 🔄
- Black elements in `.svg` `GlycoDraw` outputs are now also charcoal (`#1C1917`), just like any other format (0967d8a)
- Symbols in vertical mode are no longer rotated in `GlycoDraw`, in accordance with SNFG (0967d8a)
- `GlycoDraw` outputs now also carry metadata in notebook outputs as well as Excel cells (0967d8a)
- The `folder_filepath` argument in `plot_glycans_excel` can now also be used to specify a desired output file name (0967d8a)
- Better layouting of complex floating bits in `GlycoDraw` (0967d8a)

##### Fixed 🐛
- Fixed `Altf` and `Tagf` not being recognized as furanoses by `GlycoDraw` (0967d8a)
- Fixed `NS` not being recognized as a joint modification by `GlycoDraw` (0967d8a)
- Fixed `draw_method='chem2d', filepath='x.png'` not saving a `png` (0967d8a)

#### tokenization
##### Changed 🔄
- `get_core` and `get_modification` can now better deal with cases such as `Neu5,9Ac2` (0967d8a)
- `map_to_basic` is now cached, making functions that call it repeatedly (e.g., `structure_to_basic`) about 2x faster (e1c807a)
- `mz_to_composition` and related functions are now faster (eac80ee)

#### processing
##### Added ✨
- The LINUCS nomenclature is now also supported in Universal Input/`canonicalize_iupac` via the new `linucs_to_iupac` parser (e1c807a)
- Universal Input/`canonicalize_composition` now also supports composition nomenclatures from Byonic, FragPipe, GlycoMod, GlyHunter, GlyCombo, LaCyTools, MassyTools, GlycoGenius, and GlycReSoft (e1c807a, f936c23)

##### Changed 🔄
- `canonicalize_composition` now also accepts dictionary compositions as inputs (0967d8a)
- `canonicalize_iupac` is more robust to nomenclature variations (e1c807a)
- Moved `PDB_TO_IUPAC` from `glycontact` up into `glycowork` to facilitate atom-coloring in `.draw.draw_chem3d` without `glycontact` dependencies ()

##### Fixed 🐛
- `sanitize_iupac` no longer flags phosphodiesters (e1c807a)
- Fixed handling of variantly capitalized monosaccharides in KCF (e1c807a)
- Fixed handling of `NS` in GlycoWorkbench parsing ()

#### annotate
##### Changed 🔄
- Motif annotation is now generally faster (eac80ee)
- Motif annotations with `feature_set` `terminal` are now ~10x faster (eac80ee)

##### Fixed 🐛
- Fixed `annotate_dataset`, `annotate_glycan_topology_uncertainty`, and `get_k_saccharides` counting motifs whose stated PTM positions differ from the glycan's (e.g., `Gal6S(b1-4)GlcNAcOS` in `Gal3S(b1-4)GlcNAc6S`, or `Terminal_Glc3Ac` in glycans with `Glc6Ac` and an `Ole` residue) as soon as either side contained an `O` (9576826)

#### analysis
##### Changed 🔄
- `get_glycanova` is now faster (eac80ee)
- Every analysis function that takes a file path now also reads exports of glycomics and glycoproteomics tools (via `read_abundances`), and `get_differential_expression`/`get_glycanova` switch to glycoproteomics mode by themselves for glycoproteomics exports ()

#### graph
##### Added ✨
- Added the new `subsumes` keyword argument to `compare_glycans`, to support asymmetric wildcard matching (glycan_b is glycan_a or a more specific version of it, but not vice versa) (9576826)

##### Fixed 🐛
- Fixed `compare_glycans("Gal3S(b1-4)GlcNAcOS", "Gal6S(b1-4)GlcNAc6S")` returning `True` (PTM wildcards made specific PTMs non-comparable; same in `subgraph_isomorphism`) (9576826)

#### regex
##### Fixed 🐛
- Fixed `get_match` matching residues with differently stated PTM positions as soon as the glycan or the pattern contained a PTM wildcard (9576826)

### ml
#### inference
##### Changed 🔄
- `glycans_to_emb` can now also be used with `GIFFLAR`-type models (27dccf6)

#### models
##### Added ✨
- A trained `GIFFLAR` checkpoint is now available (for embeddings etc) via `prep_model` (f568dc2)

##### Changed 🔄
- `GIFFLAR`-type models are now more accurate and almost twice as fast to train (Kingdom prediction MCC 0.835 to 0.876, macro-F1 0.758 to 0.868; Phylum MCC 0.746 to 0.812, macro-F1 0.579 to 0.746), via two-layer GIN updates with learnable self-weights, residual updates, per-node-type pooling, linkage-aware monosaccharide edges, and 4 instead of 8 layers (27dccf6)
- `GIFFLAR` now takes a `dropout` keyword argument (27dccf6)
- Improved the `SweetNet` model architecture via BatchNorm plus residual updates after each convolution, with mean and max pooling (f568dc2)
- The trained `SweetNet` checkpoint via `prep_model` has been updated (f568dc2)

#### model_training
##### Changed 🔄
- Lowered default `rho` for `SAM` from 0.5 to 0.05 (f568dc2)

### glycan_data
#### stats
##### Added ✨
- Added `impute_biosynthetic`, a bespoke glycomics imputer that models log-abundances with per-sample activities of each biosynthetic step (monosaccharide and monosaccharide-linkage counts) plus a low-rank term, stacks it with stable-ratio partner glycans, and corrects for detection-limit censoring with a fitted selection model (f568dc2)

##### Changed 🔄
- `impute_and_normalize`, and with it every function with an `impute` keyword, now imputes with `impute_biosynthetic` instead of `MissForest` (kept only for `circadian = True`): 33-46% lower log2 RMSE than `MissForest` across 41 glycomics datasets, 12-23% lower than MICE, and 94% instead of 68% of true differential hits recovered (f568dc2)
- Cells floored because a whole group is zero no longer enter imputation as measurements (f568dc2)

#### data_entry
##### Added ✨
- Added `read_glycoproteomics`, which reads the native output of FragPipe/MSFragger-Glyco and O-Pair, pGlyco3 and pGlycoQuant, Byonic and Byologic, GlycReSoft, MetaMorpheus O-Pair, Glyco-Decipher, StrucGP, and PEAKS GlycanFinder into a `protein_site_composition` x sample `GlycoDataFrame`, ready for `get_differential_expression(glycoproteomics = True)` and `get_glycoshift_per_site` (f936c23)
- Added `read_glycomics`, which reads Skyline reports, LaCyTools and MassyTools summaries, and GlycoGenius, GlyHunter, GlycReSoft, GlycoWorkbench workspaces (.gwp) and annotated peak lists (.gwa), Thermo Compound Discoverer compound tables, and CandyCrunch tables into a glycan x sample `GlycoDataFrame`, with every label turned into a canonical sequence or composition and GlycoGenius groups kept as contrasts (f936c23, 43d7a40)