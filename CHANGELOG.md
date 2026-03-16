# Changelog

## [1.8.0]
- fixed `Quarto` accessing of `pyproject.toml` attributes for doc building (cd9b62f)

### glycan_data
#### loader
##### Added ✨
- Added new N- and O-glycomics dataset from https://pubmed.ncbi.nlm.nih.gov/41460292/ to `glycomics_data_loader` (`mouse_taysachs_N_PMID41460292` and `mouse_taysachs_O_PMID41460292`) (52c6cf9)
- Added new N-glycomics dataset from https://pubmed.ncbi.nlm.nih.gov/39877544/ to `glycomics_data_loader` (`human_serum_N_PMID39877544`) (57f6260)
- Added new N-glycomics dataset from https://pubmed.ncbi.nlm.nih.gov/37639587/ to `glycomics_data_loader` (`human_neutrophils_N_PMID37639587`)

##### Changed 🔄
- Specified wildcards in `glycomics_human_colorectal_O_PMC9254241` (e71550d)

##### Fixed 🐛
- Made sure that incomplete API access in `get_molecular_properties` does not lead to outright failure (52c6cf9)
- `glycomics_data_loader` and other `LazyLoader` instances are now robust against duplicate column names with the `.1`, `.2` suffix (they will be stripped now) (44e8473)

##### Deprecated ⚠️

### motif
#### annotate
##### Added ✨
- `get_k_saccharides` and `annotate_dataset` can now dynamically create enrichment motifs of the type `Sia(a2-3)Gal` or `Terminal_Sia(a2-3/6)` if multiple sialic acid types are present in input data (522b7cf)

##### Fixed 🐛
- Made sure curly bracket sequence content ("floaty bits") are correctly counted in `count_unique_subgraphs_of_size_k` (522b7cf)
- Make sure all narrow linkage wildcards, even if not present in `linkages`, are being correctly parsed in `count_unique_subgraphs_of_size_k` (5220912)

#### graph
##### Changed 🔄
- Added `_prefilter_labels` for more cheap checks to avoid graph operations and thus make `compare_glycans` and `subgraph_isomorphism` considerably faster (b865229)
- Made `glycan_to_graph` function much faster (up to 10x) (750cdb1)
- Made `graph_to_string_int` function ~40% faster (750cdb1)

##### Deprecated ⚠️
- Deprecated `evaluate_adjacency`; will be handled in-line in `glycan_to_graph` (750cdb1)
- Deprecated `canonicalize_glycan_graph`; will be handled in-line in `graph_to_string_int` (750cdb1)
- Deprecated `neighbor_is_branchpoint`; no longer in use (e020ffb)

#### draw
##### Changed 🔄
- `HexN`, `dHexNAc`, and `HexA` shapes now get drawn in fewer objects/more efficiently (10da7c5)

##### Fixed 🐛
- Fixed displaying beta-linkages instead of alpha-linkages in `annotate_figure` (e71550d)

#### analysis
##### Changed 🔄
- `get_volcano` can now also deal with input dataframes that have the `Glycan` column be the index instead (e71550d)
- Equivalence p-values in `get_differential_expression` now also use the same sample-size adjusted alpha as regular p-values (3884125)
- Specifying `return_plot=True` in `get_heatmap` will now also return the column names and the transformed dataframe, next to the plot object (3b72129)

##### Fixed 🐛
- CLR-transformation for paired data in `preprocess_data` now correctly uses the shared geometric mean as reference, to preserve within-pair differences (3884125)
- Fixed equivalence p-values in `get_differential_expression` if `sets=True` (3884125)
- CLR-transformed motif-level quantification in `preprocess_data` and `get_pca` used the glycan-level geometric mean as a reference, rather than the motif-level geometric mean, which is now fixed (c71c385)

#### tokenization
##### Added ✨
- `mz_to_composition` now has a new keyword argument `deprioritized`, which is a set of disfavored monosaccharides/modifications that will only be used if no composition can be found otherwise (i.e., less harsh than full exclusion via `filter_out`). This keyword argument is now also exposed in `mz_to_structures` (316f962)

#### tokenization
##### Changed 🔄
- `canonicalize_iupac` now is even more robust regarding typo correction (acf05e1)

### network
#### biosynthesis
##### Added ✨
- Added `build_network_from_glycans` handler to do a BFS-search to get the bulk biosynthetic network going (b865229)
- Added `hierarchical` option (now the new default) to the keyword argument options in `plot_format` in `plot_network`, for a more organized network display (8d03348)
- `extend_network` now has the new `auto_steps` keyword argument, which (if `to_extend` is a target composition), will calculate the minimum number of steps, cross-check it against the provided maximum as `steps`, and then iteratively extend the most favorable leaf nodes toward the target composition (f8f2fa9)

##### Changed 🔄
- `construct_network` is now more than twice as fast (a1c810c, b865229)
- Dynamic wildcard construction in `get_differential_biosynthesis` now also creates the most parsimonious narrow wildcards, similar to `annotate` (e71550d)
- Renamed the `Feature` column in `get_differential_biosynthesis` to `Glycan` (e71550d)
- `extend_network` now accepts compositions in any format in the `to_extend` keyword argument, using Universal Input (18f7ba5)
- `extend_network` now early-exits if the composition provided in `to_extend` already exists within the network, outputting the existing matching structures in the network (18f7ba5)

##### Fixed 🐛
- Fixed reaction hover label in `plot_network` (8d03348)

##### Deprecated ⚠️
- Deprecated `find_shared_virtuals`, `adjacencyMatrix_to_network`, `get_virtual_nodes`, `get_neighbors`, `create_adjacency_matrix`; now all handled in-line (a1c810c, b865229)
- Deprecated `find_path`, `find_shortest_path`, `deorphanize_nodes`, `shells_to_edges`, which is all now handled by the new `build_network_from_glycans` (b865229)