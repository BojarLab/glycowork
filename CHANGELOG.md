# Changelog

## [1.9.1]

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

##### Changed 🔄
- `get_pca` now correctly filters out redundant motifs for the PCA analysis (a918a2e)

##### Fixed 🐛
- `get_time_series` and `get_jtk` now correctly do motif quantification followed by CLR/ALR (instead of the other way around) in case of motif-analysis (a918a2e)

#### processing
##### Fixed 🐛
- Fixed `canonicalize_iupac` messing up narrow modification wildcards (e.g., `Gal3/6S`) in side branches (3a02eff)
- Fixed `canonicalize_iupac` sometimes ordering multiple floating bits wrongly (b6689a5)

#### draw
##### Changed 🔄
- Drawing "forbidden" monosaccharides, such as in `GlycoDraw("Gal(b1-3)[!GlcNAc(b1-6)]GalNAc")`, now automatically makes the forbidden monosaccharides and their linkages transparent (3621350)

#### graph
##### Fixed 🐛
- Fixed `subgraph_isomorphism` counting too many isomorphic matches in certain scenarios of linkage ambiguity (81244c9)
- `get_linkage_number` now always handles narrow linkage wildcards correctly for branch sorting purposes (d222f41)

#### annotate
##### Added ✨
- Added `get_motif_dag` to build a containment DAG of connected motifs, for comparative glycomics analyses in `analysis` (a918a2e)

##### Changed 🔄
- Refined counting of `Terminal_` motifs in `annotate_dataset` (a918a2e)

### network
#### biosynthesis
##### Changed 🔄
- In `construct_network`, `edge_type=enzyme` will now assign glycan class-specific enzymes, if possible (e.g., only ST3GAL4 for N-glycans instead of all ST3GALs) (cb262b3)

### glycan_data
#### stats
##### Added ✨
- `clr_transformation` now has a new `reference` keyword argument, to optionally specify from which variables the geometric mean should be constructed (a918a2e)

##### Changed 🔄
- `hotellings_t2` is now more robust to tiny groups (a918a2e)
- `TST_grouped_benjamini_hochberg` is now more robust to groups with π0 = 1