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

### network
#### biosynthesis
##### Changed 🔄
- `construct_network` is now more than twice as fast

##### Deprecated ⚠️
- Deprecated `find_shared_virtuals` and `adjacencyMatrix_to_network`; now handled in-line in `create_adjacency_matrix`