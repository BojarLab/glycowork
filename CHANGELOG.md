# Changelog

## [1.10.1]
- Bumped `glycorender` requirement to `0.4.1` to support `GlycoDraw` die cut stickers (257ba7b)
- Added more error messages throughout the package (4f9ae4c)

### glycan_data
#### loader
##### Added ✨
- Added new curated glycomics datasets to `glycomics_data_loader`: `human_influenza_N_10_648982026_02_02_703422`, `human_influenza_O_10_648982026_02_02_703422` (f91d6b4)

### motif
#### tokenization
##### Added ✨
- Added permethylated + peracetylated masses for `HexN`, `PCho`, and `PEtN` (df1cd48)

##### Changed 🔄
- `compositions_to_structures`, `prot_to_coded`, and `constrain_prot` now also accept single string inputs in addition to lists (60b3d6d)

##### Fixed 🐛

##### Deprecated ⚠️

#### analysis
##### Added ✨
- Added `get_distance_matrix` to compute sample-sample or glycan/motif-glycan/motif distance matrices directly from abundance dataframes (9e99f25)
- Added the new `dist_func` keyword argument to `get_heatmap` and `get_biodiversity`, to customize the distance function used for clustering (9e99f25)
- Added the new `get_pcoa` function to perform principal coordinate analysis (PCoA) (9e99f25)

##### Changed 🔄
- `annotate_figure` in `get_volcano` is now default-True instead of default-False (f91d6b4)
- Changed the output type of `get_biodiversity`'s beta distance matrix from `ndarray` to `pd.DataFrame` (9e99f25)

#### draw
##### Added ✨
- `GlycoDraw` and `display_svg_with_matplotlib` have received the new `sticker` keyword argument, to output die cut sticker style drawings of glycans (257ba7b)
- `GlycoDraw` can now also draw compositions, such as `GlycoDraw("H3N4A2")`, supporting all composition formats via Universal Input (81ad77a)

##### Changed 🔄
- `get_mono_atoms` now also accepts single string inputs in addition to lists (60b3d6d)

##### Fixed 🐛
- Fixed display of floating sulfate positional indicators ()

#### annotate
##### Fixed 🐛
- Fixed `get_k_saccharides` outputting disaccharides if `size` > 2 and `up_to=False` (a523286)

##### Changed 🔄
- `get_molecular_properties`, `annotate_dataset`, `get_k_saccharides`, `get_size_branching_features`, and `get_minimal_ksaccharide_ambiguity` now also accept single string inputs in addition to lists (60b3d6d)

### processing
##### Changed 🔄
- `glytoucan_to_glycan`, `expand_lib`, and `get_lib` now also accept single string inputs in addition to lists (60b3d6d)
- Made sure `min_process_glycans` is robust to empty inputs ()

### network
#### biosynthesis
##### Added ✨
- Added more literature-known enzyme constraints to make `construct_network` outputs more physiological if `constraints=True` (573cf04)
- Added more literature-known substrate specificities to `monolink_to_enzyme` for more physiological edge labeling in `plot_network` (c4910e9)

##### Changed 🔄
- `draw_glycans` in `plot_network` is now default-True instead of default-False (f91d6b4)
- `construct_network` can now also use dataframes with a glycan column as input (in addition to the standard list of glycans) (4791cc1)
- `get_maximum_flow` and `get_max_flow_path` no longer default to milk oligosaccharide roots in `source` but rather infer the default root from the provided network (4791cc1)
- Acetylated sialic acids are now considered as one transfer in `construct_network` etc, rather than sialylation followed by acetylation (since CASD1 acetylates CMP-Neu5Ac etc) (60b3d6d)
- `construct_network`, `choose_path`, `trace_diamonds`, `evoprune_network`, `infer_network`, and `extend_glycans` now also accept single string inputs in addition to lists (60b3d6d)

##### Fixed 🐛
- Fixed `extend_network(prioritize=True)` handling of multi-root networks (a5121ae)
- Fixed auto-inferring of group contrasts in `get_differential_biosynthesis` if `longitudinal=True` (3e0133b)

### ml
#### inference
##### Changed 🔄
- `get_esmc_representations`, `glycans_to_emb`, `get_Nsequon_preds`, and `get_lectin_preds` now also accept single string inputs in addition to lists (60b3d6d)