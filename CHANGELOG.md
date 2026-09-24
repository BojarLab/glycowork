# Changelog

## [1.10.1]
- Bumped `glycorender` requirement to `0.4.2` to support `GlycoDraw` die cut stickers, 300 dpi PNG export, and upright linkage labels (257ba7b, )
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
- `composition_to_mass` now raises an error for an unknown reducing-end `modification` instead of silently ignoring it (4f9ae4c)
- `tolerance_unit` in `mz_to_composition` is now case-insensitive (4f9ae4c)

#### analysis
##### Added ✨
- Added `get_distance_matrix` to compute sample-sample or glycan/motif-glycan/motif distance matrices directly from abundance dataframes (9e99f25)
- Added the new `dist_func` keyword argument to `get_heatmap` and `get_biodiversity`, to customize the distance function used for clustering (9e99f25)
- Added the new `get_pcoa` function to perform principal coordinate analysis (PCoA) (9e99f25)

##### Changed 🔄
- `annotate_volcano` in `get_volcano` is now default-True instead of default-False (f91d6b4)
- Changed the output type of `get_biodiversity`'s beta distance matrix from `ndarray` to `pd.DataFrame` (9e99f25)

##### Fixed 🐛
- Fixed `get_heatmap` placing its axis labels on the colorbar instead of the heatmap (4f9ae4c)
- Fixed `get_meta_analysis` crashing when saving a forest plot without `study_names` (4f9ae4c)
- Fixed `get_volcano(annotate_volcano = True)` showing nothing outside of Jupyter when no `filepath` is given ()

#### draw
##### Added ✨
- `GlycoDraw` and `display_svg_with_matplotlib` have received the new `sticker` keyword argument, to output die cut sticker style drawings of glycans (257ba7b)
- `GlycoDraw` can now also draw compositions, such as `GlycoDraw("H3N4A2")`, supporting all composition formats via Universal Input (81ad77a)

##### Changed 🔄
- `get_mono_atoms` now also accepts single string inputs in addition to lists (60b3d6d)
- `GlycoDraw` now warns if `highlight_motif` does not occur in the glycan (4f9ae4c)
- SVGs saved by `GlycoDraw` now match its PDF/PNG output: linkage labels read upright (also in vertical mode), modifications are bold, furanose *f* is italic, and linkages have no gaps ()
- `GlycoDraw` PNG exports are now rendered at 300 dpi at the same physical size as the PDF, and Jupyter previews at 2x for crisp display on high-DPI screens ()

##### Fixed 🐛
- Fixed display of floating sulfate positional indicators (e6971ee)
- Fixed `annotate_figure` (and thereby `get_volcano`) crashing on glyco-regex motifs such as `Nglycan_hybrid`, which now stay text labels ()

#### annotate
##### Fixed 🐛
- Fixed `get_k_saccharides` outputting disaccharides if `size` > 2 and `up_to=False` (a523286)
- Fixed `get_glycan_similarity` failing on graph inputs (4f9ae4c)

##### Changed 🔄
- `get_molecular_properties`, `annotate_dataset`, `get_k_saccharides`, `get_size_branching_features`, and `get_minimal_ksaccharide_ambiguity` now also accept single string inputs in addition to lists (60b3d6d)
- `annotate_dataset`, `annotate_glycan`, and `annotate_glycan_topology_uncertainty` now also accept motif dataframes without a `termini_spec` column (4f9ae4c)

#### graph
##### Changed 🔄
- `subgraph_isomorphism` and `expand_termini_list` now also accept a single string as `termini_list`, and raise an error for unknown entries (4f9ae4c)

#### regex
##### Changed 🔄
- `get_match` now also reads IUPAC-short input such as `Galb1-4GlcNAc`, and `get_match_batch` also accepts a single glycan (4f9ae4c)

#### processing
##### Changed 🔄
- `glytoucan_to_glycan`, `expand_lib`, and `get_lib` now also accept single string inputs in addition to lists (60b3d6d)
- Made sure `min_process_glycans` is robust to empty inputs (e6971ee)
- `canonicalize_composition` now also reads five-field (Hex/HexNAc/Neu5Ac/Neu5Gc/dHex) underscore- and whitespace-separated compositions (3e0133b)
- `parse_glycoform` now accepts compositions in any format supported by `canonicalize_composition` (4f9ae4c)
- `oxford_to_iupac` now also reads names such as `Man9GlcNAc2` (4f9ae4c)
- `max_specify_glycan` now also accepts species names with spaces instead of underscores (4f9ae4c)

##### Fixed 🐛
- Fixed `canonicalize_iupac` silently turning compositions such as `H9N2` or `H3N3S1` into bare N-glycan cores by misreading them as Oxford nomenclature; they are now returned unchanged ()

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