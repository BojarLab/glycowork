# Changelog

## [1.6.0]
- All glycan graphs are now directed graphs (`nx.Graph` --> `nx.DiGraph`), flowing from the root (reducing end) to the tips (non-reducing ends), which has led to code changes in quite few functions. Some functions run faster now, yet outputs are unaffected (03dfad6)
- Added `huggingface_hub>=0.16.0` as a new dependency to facilitate more robust model distribution (22f6b8f)
- Moved `drawSvg~=2.0` and `openpyxl` from the optional `[draw]` install to the dependencies of base glycowork. That allows for the usage of `GlycoDraw` in, e.g., Jupyter environments etc, even if `glycowork[draw]` has not been installed. Since these dependencies are unproblematic, no special install needs to be followed for the base glycowork install (60e51da)
- Deprecated the optional `[draw]` install completely, by replacing the problematic `cairosvg` dependency with our new & custom renderer `glycorender`, which is now a new base dependency of `glycowork` (7c4fbe1)
- Moved `Pillow` dependency into `glycorender` (793e71f)
- Deprecated `mpld3` and `matplotlib-inline` dependencies; added new `bokeh` and `IPython` base dependencies for better interactive plotting in a Jupyter environment (972c34b, 13b0699)
- Formally added `numpy` and `matplotlib` to base dependencies (ba40c73)
- Exposed `canonicalize_iupac` to the `glycoworkGUI` (ba40c73)
- Implemented submodule lazy loading to speed up package imports & start-up (9bf18f7)

## glycan_data
#### loader
##### Added ✨
- Added `HashableDict` class to allow for caching of functions with dicts as inputs (03dfad6)
- Added `GlycoDataFrame` class to extend `pd.DataFrame` by adding the `.glyco_filter` method, to easily filter glycan dataframes by the occurrence/count of sequence motifs (9764b3e)
- Added new curated glycoproteomics dataset: `sorghum_N_PMID39137587` (13b0699)
- Updated `glycan_binding`, `df_glycan`, `df_species` to be bigger, better, and cleaner (e302075)

##### Changed 🔄
- Refined motif definition of `Internal_LewisX`/`Internal_Lewis_A`/`i_antigen` in `motif_list`, to exclude `LewisY`/`LewisB`/`I_antigen` from matching/overlapping (07c9c12)
- Renamed `Hyluronan` in `motif_list` into `Hyaluronan` (07c9c12)
- Removed `Nglycolyl_GM2` from `motif_list`; it's captured by `GM2` (07c9c12)
- Further curated glycomics datasets stored in `glycomics_data_loader` by introducing the b1-? --> b1-3/4 narrow linkage ambiguities (9eeaa3a, 436bf09)
- `download_model` will now download model weights and representations from the HuggingFace Hub (22f6b8f)
- `df_species` and `df_glycan` are now of type `GlycoDataFrame`; `build_custom_df` now returns a dataframe of type `GlycoDataFrame` (9764b3e)
- `DataFrameSerializer` will now also correctly serialize cells in which (i) lists of strings or (ii) dictionaries have been converted into one string (Excel/pandas interplay of complex cells), where we use `ast` to try to literally evaluate them back into lists of strings (i) / dictionaries (ii) (806a47c, e302075)

#### stats
##### Fixed 🐛
- Fixed a `DeprecationWarning` about implicit indexing in `alr_transformation` when a dict is used for `custom_scale` (9bf18f7)

### motif
#### processing
##### Added ✨
- GlyTouCanIDs are now another supported nomenclature in the context of Universal Input and can be used as inputs for functions etc, supported via improvements in `canonicalize_iupac` (eafb218)
- Added `sanitize_iupac` to detect and fix chemical impossibilities (like two monosaccharides connected via the same hydroxyl group) and fix it (407cd6f, 74d35a0)
- Added `GLYCAN_MAPPINGS` dictionary to map commonly used glycan names to their IUPAC-condensed sequence (36d33b8)
- Added `linearcode1d_to_iupac` to support sequences of type `01Y41Y41M(31M21M21M)61M(31M21M)61M21M` in the Universal Input platform (d0eee40)
- CSDB linear code is now another supported nomenclature in the context of Universal Input and can be used as inputs for functions etc, supported via improvements in `canonicalize_iupac` (8dd34b7, 36d2a61, 69c00e1, 2d8fdfd, cb97593)
- Added `transform_repeat_glycan` to support bringing repeat structures of type `1)Fruf(b2-3)Fruf(b2-` into the glycowork format of `Fruf(b2-3)Fruf(b2-1)Fruf` (36d2a61, 2d8fdfd)
- Added `nglycan_stub_to_iupac` to support sequences of type `(Hex)3 (HexNAc)1 (NeuAc)1 + (Man)3(GlcNAc)2` in the Universal Input platform (69c00e1)
- Added `iupac_to_smiles` alias for `IUPAC_to_SMILES` (cb97593)
- Added `GAG_disaccharide_to_iupac` to support disaccharide structural code (DSC) for GAGs (e.g., `D2A6`) in the context of Universal Input (0770bcd)
- Added more WURCS tokens for better support in the context of Universal Input, now stored in `wurcs_tokens.json` (b30553f, b94cf6d, d1fd4c7, 14bbd4d, a109176)
- Support monosaccharides without anomeric indicator and phospho-linkages in WURCS (14bbd4d)

##### Changed 🔄
- Moved `.motif.query.glytoucan_to_glycan` into `.motif.processing` (eafb218)
- `canonicalize_iupac` will now use `sanitize_iupac` to auto-fix chemical impossibilities in input glycans (407cd6f)
- More GlycoWorkBench sequence variants can now be handled via `glycoworkbench_to_iupac`/`canonicalize_iupac` (9eeaa3a, 436bf09, 74d35a0)
- More supported WURCS2 tokens (436bf09, 84c5bcc)
- `canonicalize_iupac` and most glycowork functions now also support common names, like "LacNAc" or "2'-FL", in the Universal Input framework, thanks to `GLYCAN_MAPPINGS` (36d33b8, ab42dbb)
- `get_class` can now identify repeating unit glycans and returns "repeat" in this case (74d35a0)
- `canonicalize_iupac` can now handle even more IUPAC-dialects, like `aMan13(aMan16)Man`, where the anomeric state is declared before the monosaccharide (24c8e81, ab42dbb)
- `canonicalize_iupac` will now use `glycan_to_nxGraph` and `graph_to_string` for branch canonicalization, instead of `choose_correct_isoform`. On average, this works much better and is more reliable (7c52a0e)
- `canonicalize_iupac` is now more robust to (5-6) type linkages and to the associated sugar alcohols, like Rib5P-ol (7a260ac)
- `canonicalize_iupac` will now raise a `ValueError` instead of a warning if a glycan string has mismatching brackets (b69fced)
- `canonicalize_iupac` can now handle even more IUPAC-dialects such as `Neu5Ac-α-2,6-Gal-β-1,3-GlcNAc-β-Sp` (cb2c898)
- `get_class` will now correctly annotate plant N-glycans with core a1-3 Fuc (8dd34b7)

##### Deprecated ⚠️
- Deprecated `find_isomorphs` and `choose_correct_isoform`; this will be done (and better) by the new `canonicalize_glycan_graph` instead (7c52a0e)

#### annotate
##### Changed 🔄
- Renamed `clean_up_heatmap` to `deduplicate_motifs` (407cd6f)
- Allow sets of glycans as inputs in `get_k_saccharides`, in addition to lists of glycans (74d35a0)
- Made `get_k_saccharides` faster by re-using graphs and using the directed graphs in an optimized way (7c52a0e)
- `get_terminal_structures` will now return an actual `ValueError` when setting `size` to be higher than 2 (fa451ba)

##### Fixed 🐛
- Fixed an edge case in `get_k_saccharides`, in which choosing a `size` larger than the size of the largest glycan in the input caused an error (db7847d)
- Fixed `get_k_saccharides` with higher values of `size`, which occasionally produced invalid strings, by refactoring `count_unique_subgraphs_of_size_k` and switching it to use the changed `graph_to_string_int`, to ensure motif validity (db7847d)
- Fixed `preprocess_data`, which was attempting to transform 0-containing dataframes when no transform argument was provided (878701a)

##### Deprecated ⚠️
- Deprecated `link_find`; will be done by an optimized `get_k_saccharides` instead (since `link_find` relied on `find_isomorphs`) (7c52a0e)

#### draw
##### Added ✨
- Added `get_branches_from_graph` to process directed glycan graphs into components for `GlycoDraw` (e56d015)
- Added the `reverse_highlight` keyword argument to `GlycoDraw`, if you want to highlight everything *except* a certain motif (which means you can highlight discontiguous sequence stretches) (f5e3b2f)
- `GlycoDraw` will now inject ALT text / metadata into all its outputs (displayed or saved as `.pdf`/`.svg`/`.png`) for improved accessibility and to aid curation efforts. The ALT text will be automatically generated and includes appropriate tags, the glycan sequence, and used drawing options. But it can also be overriden, if desired, via the new `alt_text` keyword argument in `GlycoDraw` (793e71f)

##### Changed 🔄
- Quantitative highlighting in `GlycoDraw` via the `per_residue` keyword argument will now use individual SNFG-colors instead of a uniform highlight color (07c9c12)
- Refactored `get_coordinates_and_labels` to be more efficient and generalizable; with this and the new `get_branches_from_graph`, `GlycoDraw` is now capable of drawing even more complex structures accurately (e56d015, 36fbba9)
- Next to `.svg` and `.pdf`, it is now also possible to save `.png` files with `GlycoDraw` (36fbba9)
- `display_svg_with_matplotlib` now has the optional `chem` keyword argument to alert our renderer that the .svg comes from RDKit (7c4fbe1)

##### Deprecated ⚠️
- Deprecated `split_node`, `unique`, `get_indices`, `split_monosaccharide_linkage`, and `glycan_to_skeleton`, since this will now be handled by the changed `get_coordinates_and_labels` (e56d015)

#### graph
##### Added ✨
- Added `canonicalize_glycan_graph` to reorder graph nodes in either a length-first or linkage-first manner (7c52a0e)
- `graph_to_string` and its sub-functions now have new keyword arguments: `canonicalize`, to determine whether transcribed graphs should be re-ordered into a canonicalized IUPAC-condensed and `order_by` to decide whether canonicalization happens in a length-first or linkage-first manner (7c52a0e)
- Added `glycan_graph_memoize` decorator to cache results from `graph_to_string_int` (db7847d)

##### Changed 🔄
- Switched `lru_cache` from `glycan_to_graph` to `glycan_to_nxGraph_int` for better performance and fewer opportunities to mess with the cache (03dfad6)
- Made `graph_to_string` faster, to accommodate its more central role in the Universal Input framework (7c52a0e)
- Added a fast-return for disaccharide graphs in `graph_to_string_int`, since no canonicalization/branch sorting is needed (db7847d)
- `subgraph_isomorphism` is now also fine with people prodiving a separate `termini_list` even when providing graphs as input (though it's still recommended to just input `termini_list` when creating the graphs in the first place) (7a260ac)
- `GlycoDraw` will now properly space text within monosaccharide symbols, if there are multiple indicators (like `D` and `f`) (03e502c)
- `get_possible_topologies` will now raise a `ValueError` instead of a warning when you attempt to use it with an already defined glycan (b69fced)
- `get_possible_topologies` will now by default return the strings of glycans, except if `return_graphs=True`, in which case the old behavior of returning glycan graphs as NetworkX objects is restored (b69fced)
- If `get_possible_topologies` is used with dangling modifications (e.g., `{OS}Gal(b1-3)GalNAc`), the new default is now to also try adding the modification at non-terminal residues, even if `exhaustive=False`. The behavior for monosaccharide additions is unchanged. (b69fced)
- If a glycan string is input into `graph_to_string` (even though you shouldn't do this) it will simply be returned as the string value (b69fced)

##### Fixed 🐛
- Fixed an edge case in `compare_glycans` in which two identical string glycans returned (True, True) if `return_matches == False` (03dfad6)

#### tokenization
##### Changed 🔄
- `stemify_glycan` can now deal with even more strongly modified glycans and should be faster too (03e502c)
- `map_to_basic` can now deal with any linkage, even those never before seen in glycans (069faf7)

### network
#### biosynthesis
##### Changed 🔄
- `plot_network` now uses `bokeh` for interactive plotting instead of `mpld3`; changed the default layout algorithm from `kamada_kawai` to `spring` (972c34b)
- `find_diamonds` and `highlight_network` will now raise actual `ValueError`s instead of printed warnings, if their settings are wrong (36d2a61) 

##### Fixed 🐛
- Fixed a `DeprecationWarning` about `resources.open_text` in `construct_network` (ba40c73)
- Fixed an edge case in `find_diff` when related but dissimilar glycans are used as input (069faf7)

#### evolution
##### Changed 🔄
- `distance_from_metric` will now raise a `ValueError` if the chosen metric is not yet supported (ba40c73)

##### Fixed 🐛
- Fixed a `ClusterWarning` about distance matrix formats in `dendrogram_from_distance` (ba40c73)

### ml
#### model_training
##### Changed 🔄
- `training_setup` will now raise a `ValueError` if the chosen mode is not supported (2d8fdfd)

#### inference
##### Changed 🔄
- `get_lectin_preds` will now raise a `ValueError` if no protein:ESM-1b dictionary is provided in non-flex mode (9bf18f7)
- `get_esm1b_representations` is now `get_esmc_representations`, with a slightly changed function signature (e.g., no `alphabet` needed anymore) (e302075)

#### models
##### Changed 🔄
- `init_weights` will now raise a `ValueError` if the chosen initialization `mode` is not supported (9bf18f7)
- `LectinOracle` will now use ESMC-300M representations, rather than ESM-1b (e302075)