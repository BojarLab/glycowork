# Changelog

## [1.7.2]
- fixed `Quarto` accessing of `pyproject.toml` attributes for doc building (cd9b62f)

### glycan_data
#### loader
##### Added ✨
- Added new N- and O-glycomics dataset from https://pubmed.ncbi.nlm.nih.gov/41460292/ to `glycomics_data_loader` (`mouse_taysachs_N_PMID41460292` and `mouse_taysachs_O_PMID41460292`) (52c6cf9)

##### Changed 🔄

##### Fixed 🐛
- Made sure that incomplete API access in `get_molecular_properties` does not lead to outright failure (52c6cf9)

##### Deprecated ⚠️

### motif
#### annotate
##### Added ✨
- `get_k_saccharides` and `annotate_dataset` can now dynamically create enrichment motifs of the type `Sia(a2-3)Gal` or `Terminal_Sia(a2-3/6)` if multiple sialic acid types are present in input data (522b7cf)

##### Fixed 🐛
- Made sure curly bracket sequence content ("floaty bits") are correctly counted in `count_unique_subgraphs_of_size_k` (522b7cf)

#### graph
##### Changed 🔄
- Added `_prefilter_labels` for more cheap checks to avoid graph operations and thus make `compare_glycans` and `subgraph_isomorphism` considerably faster (b865229)
- Made `glycan_to_graph` function much faster (up to 10x) (750cdb1)
- Made `graph_to_string_int` function ~40% faster (750cdb1)

##### Deprecated ⚠️
- Deprecated `evaluate_adjacency`; will be handled in-line in `glycan_to_graph` (750cdb1)
- Deprecated `canonicalize_glycan_graph`; will be handled in-line in `graph_to_string_int` (750cdb1)

### network
#### biosynthesis
##### Added ✨
- Added `build_network_from_glycans` handler to do a BFS-search to get the bulk biosynthetic network going (b865229)
- Added `hierarchical` option (now the new default) to the keyword argument options in `plot_format` in `plot_network`, for a more organized network display

##### Changed 🔄
- `construct_network` is now more than twice as fast (a1c810c, b865229)

##### Fixed 🐛
- Fixed reaction hover label in `plot_network`

##### Deprecated ⚠️
- Deprecated `find_shared_virtuals`, `adjacencyMatrix_to_network`, `get_virtual_nodes`, `get_neighbors`, `create_adjacency_matrix`; now all handled in-line (a1c810c, b865229)
- Deprecated `find_path`, `find_shortest_path`, `deorphanize_nodes`, `shells_to_edges`, which is all now handled by the new `build_network_from_glycans` (b865229)