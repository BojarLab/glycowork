# Changelog

## [1.5.0]

### Added ✨
- Added type hints to all functions (e6721a1)
- Added CodeCov shield to track PyTest test code coverage (23d6456)
- Added more PyTest unit tests (e.g., 0c94995, 23d6456, 5a99d6b, f76535e, 94646ad, d5f5d4e, 918d18f, d1a8c6d, 194f31c)
- Added setuptools to required_installs to support pip installation beyond `pip 25.0` (94646ad)
- Added pyproject.toml to support pip installation beyond `pip 25.0` (94646ad)
- Added CITATION.bib to allow for even easier citation of glycowork (a64f694)

### Changed 🔄
- Bumped minimum supported Python version to 3.9 (3.8 is no longer supported, see https://devguide.python.org/versions/) (4960c5c)
- Switched docstring style to docments (<https://nbdev.fast.ai/tutorials/best_practices.html#document-parameters-with-docments>) (e6721a1)
- Removed `gdown` dependency; Will be handled by the standard library module `urllib` for better retrieval of externally stored models/files (319981e, 35ed71a)
- Switched pathing from `os` to `pathlib` (319981e)

### glycan_data
##### Added ✨
- Added new named motifs to `motif_list`: DisialylLewisC, `Sia(a2-3)Gal(b1-3)[Sia(a2-6)]GlcNAc`; RM2, `Sia(a2-3)[GalNAc(b1-4)]Gal(b1-3)[Sia(a2-6)]GlcNAc`; DisialylLewisA, `Sia(a2-3)Gal(b1-3)[Fuc(a1-4)][Sia(a2-6)]GlcNAc` (a64f694)
- Added new curated glycomics dataset, `mouse_brain_GSL_PMID39375371` (b94744e)

##### Changed 🔄
- Changed `glycoproteomics_human_keratinocytes_PMID37956981` to `glycoproteomics_human_keratinocytes_N_PMID37956981` (d5f5d4e)
- Improved the description of blood group motifs in `motif_list` (including type 3 blood group antigens, ExtB, and parent motifs) (b94744e)

##### Fixed 🐛
- Fixed the "Oglycan_core6" motif definition in `motif_list` to no longer overlap with core 2 structures (f394bda)

#### loader
##### Added ✨
- Added `count_nested_brackets` helper function to monitor level of nesting in glycans (41bb1a1, d57b836)
- Added dictionaries with lists of strings as values as a new supported data type for `DataFrameSerializer` (034b6ad)
- Added `share_neighbor` helper function to check whether two nodes in a glycan graph share a neighbor (f394bda)

##### Changed 🔄
- Changed `resources.open_text` to `resources.files` to prevent `DeprecationWarning` from `importlib` (0c94995)
- `lectin_specificity` now uses our custom `DataFrameSerializer` and is stored as a .json file rather than a .pkl file, to improve long-term stability across versions (034b6ad)

##### Fixed 🐛
- Fixed DeprecationWarning in all data-loading functions that used `importlib.resources.open_text` or `.content` (87ea2fc)

#### stats
##### Added ✨
- Added the "random_state" keyword argument to `clr_transformation` to allow users to provide a reproducible RNG seed (b94744e)
- Added the `JTKTest` class object (87ea2fc)

##### Changed 🔄
- For `replace_outliers_winsorization`, in small datasets, the 5% limit is dynamically changed to include at least one datapoint (23d6456)
- Handled the edge case of strong differences in `cohen_d` with zero standard deviation; now outputting positive/negative infinity (23d6456)
- Renamed `test_inter_vs_intra_group` to `compare_inter_vs_intra_group`, to avoid testing issues (23d6456)
- `partial_corr` will now return a normal Spearman's correlation if no control features can be identified (241141b)

##### Deprecated ⚠️
- Deprecated `hlm`, `fast_two_sum`, `two_sum`, `expansion_sum`, and `update_cf_for_m_n`, which will all be done in-line instead (e1afe33)
- Deprecated `jtkdist`, `jtkinit`, `jtkstat`, `jtkx`, which will all be done by the new `JTKTest` (87ea2fc)

##### Fixed 🐛
- Fixed DeprecationWarning in `calculate_permanova_stat` for calling nonzero on 0d arrays (23d6456)
- Prevent possible division by zero in pseudo-F calculation in `calculate_permanova_stat` (23d6456)
- Fixed DeprecationWarning in `jtkdist` for calling `np.sum(generator)` (23d6456)
- Ensured that the input to `impute_and_normalize` are columns with floats, to avoid TypeWarnings during imputation (23d6456)
- Fixed DeprecationWarning in `process_glm_results` to prevent DataFrameGroupBy.apply from operating on the grouping columns (23d6456)
- Fixed RuntimeWarnings for JTK-related functions in case of imperfect input data (d5f5d4e)
- Ensured that `correct_multiple_testing` will return empty lists if the provided p-value list is also empty (ef3da9c)

### motif
#### tokenization
##### Added ✨
- Added `get_random_glycan` to retrieve random glycan sequences (optionally of specific glycan type) (d1a8c6d)
- Supported intramolecular modifications like lactonization in `glycan_to_composition` (8c69c2c)

##### Changed 🔄
- Changed `resources.open_text` to `resources.files` to prevent `DeprecationWarning` from `importlib` (0c94995)
- The monosaccharide keys of the output dictionaries of `glycan_to_composition` are now alphabetically sorted (8c69c2c)
- Modified `calculate_adduct_mass` to deal with a greater variety of adduct handling, such as "C2H4O2", "-H2O", "+Na" to add or subtract masses (8c69c2c)
- Expanded `glycan_to_mass` and `composition_to_mass` to deal with compositional building blocks that represent losses/gains in the molecule (like "-H2O") (8c69c2c)
- Composition and mass functions now can correctly work with azide-modified monosaccharides such as Neu5Az (ef3da9c)
- In addition to chemical formulae, users can now also provide direct additional masses as floats with the same "adduct" keyword argument in `composition_to_mass` and `glycan_to_mass` (d57b836)
- `get_modification` will no longer return the 5Ac / 5Gc of Neu5Ac / Neu5Gc as part of the modification (0387d37)

##### Fixed 🐛
- Fixed an edge case in `get_unique_topologies`, in which the absence of a universal replacer sometimes created an empty list that was attempted to be indexed (0c94995)
- Made sure that `compositions_to_structures` always returns a DataFrame, even if no matches are found (0c94995)
- Provided correct exact methyl masses in `mass_dict` (e3eeb32)

#### processing
##### Added ✨
- Added "antennary_Fuc" as another inferred feature to `infer_features_from_composition` (a64f694)
- Added "IdoA", "GalA", "Araf", "D-Fuc", "AllNAc", "Par", "Kdo", "GlcN", "Ido", "Col", "Tyv", "GalN", "QuiNAc", "Gul", and "Gal6S" to recognized WURCS2 tokens (52fc16e, f3cd8f0, 7551805, 35ed71a)
- Added the new "order_by" keyword argument to `choose_correct_isoform` to enforce strictly sorting branches by branch endings / linkages, if desired (918d18f)
- Added "Col", "Ido", "Kdo", and "Gul" to supported GlycoCT monosaccharides (7551805, 35ed71a)
- GLYCAM is now another supported nomenclature in the Universal Input framework, enabled by the added `glycam_to_iupac` function, which is also integrated into `canonicalize_iupac` (2fb5dc6)
- GlycoWorkBench (GlycanBuilder) is now another supported nomenclature in the Universal Input framework, enabled by the added `glycoworkbench_to_iupac` function, which is also integrated into `canonicalize_iupac` (ea1fdfc)

##### Changed 🔄
- `check_nomenclature` will now actually raise appropriate Exceptions, in case nomenclature is incompatible with glycowork, instead of print warnings (23d6456)
- Supported triple-branch reordering in `find_isomorphs` and `choose_correct_isoform` (918d18f)
- Improved `find_isomorphs` to swap neighboring branches with different levels of nesting (41bb1a1, 034b6ad)
- `choose_correct_isoform` can now also be used with a single glycan sequence, in which case it internally calls `find_isomorphs` to generate material for choosing (918d18f)
- `choose_correct_isoform` can now correctly handle more complex sequences than before (41bb1a1, 034b6ad)
- `canonicalize_iupac` now can handle modifications such as Neu5,9Ac2 / Neu4,5Ac2 or multiple ones like in (6S)(4S)Gal, even if in the wrong order (034b6ad)
- `canonicalize_iupac` now can handle even more typos (e.g., 'aa1-3' in specifying a linkage) (a64f694, 241141b)
- `canonicalize_iupac` now can handle even more inconsistencies (e.g., mix of short-hand and expanded linkages)
- Expanded `get_mono` to deal with some special WURCS2 tokens at the reducing end, of type `u2122h_2*NCC/3=O` (d57b836)
- `canonicalize_iupac` will no longer convert things like "b1-3/4" into "b1-?", because narrow linkage ambiguities can now be properly handled (52fc16e)
- `get_possible_linkages` and `de_wildcard_glycoletter` now also support narrow linkage ambiguities like "b1-3/4" (52fc16e)
- `canonicalize_iupac` will now no longer mess up branch formatting of the repeating unit in glycans of type "repeat" (9a94537)
- Ensured that `canonicalize_iupac` works with lactonized glycans (i.e., containing something like "1,7lactone") (8c69c2c)
- `find_matching_brackets_indices` has been renamed to `get_matching_indices` and now takes multiple delimiter choices and returns a generator, including the level of nesting (basically what `.draw.matches` used to do) (e1afe33)
- `get_class` will now return "lipid/free" if glycans of type Neu5Ac(a2-3)Gal(b1-4)Glc are supplied (i.e., lacking 1Cer and -ol but still lactose-core based) (b99699c)
- `expand_lib` now no longer modifies the input dictionary (65bd12c)
- `get_possible_linkages` now returns a set instead of a list (a98461f)
- `wurcs_to_iupac` now can also properly deal with ultra-narrow linkage wildcards (e.g., a2-3/6) (f3cd8f0)

##### Fixed 🐛
- Fixed component inference in `parse_glycoform` in case of unexpected composition formats (0c94995)
- Fixed an issue in `equal_repeats`, in which identical repeats sometimes were not returning True (0c94995)

#### graph
##### Added ✨
- Natively support narrow linkage ambiguity in `categorical_node_match_wildcard`; that means you can use things like "Gal(b1-3/4)GlcNAc" with `subgraph_isomorphism` or `compare_glycans` (as well as all functions using these core functions) and it will only return True for "Gal(b1-3)GlcNAc", "Gal(b1-4)GlcNAc", and "Gal(b1-?)GlcNAc" (b94744e)
- Added `build_wildcard_cache` for a central handling of wildcard mapping that can also be cached (a98461f)
- `compare_glycans` now also has the `return_matches` keyword argument that allows for a retrieval of the node mapping if the glycans are isomorphic (7c510c9)

##### Changed 🔄
- Ensured that `compare_glycans` is 100% order-specific, never matching something like ("Gal(b1-4)GlcNAc", "GlcNAc(b1-4)Gal") (5a99d6b)
- `glycan_to_nxGraph` will now return an empty graph if the input is an empty string (4f1ccfa)
- `get_possible_topologies` will now also produce a warning (and return the input) if an already defined topology is provided as a pre-calculated graph (3f22f14)
- Negation in `subgraph_isomorphism` can now also be added for internal monosaccharides (e.g., "Neu5Ac(a2-3)!Gal(b1-4)GlcNAc") (7558d9b)
- Functions with the `handle_negation` decorator can now be accessed without the decorator via `.__wrapped__` (7558d9b)

##### Fixed 🐛
- Fixed an edge case in which `subgraph_isomorphism` could erroneously return False if any of the matchings were in the wrong order, if "count = False" (f394bda)
- Fixed an edge case in which negated motifs in `subgraph_isomorphism` sometimes wrongly returned False because the negated motif was present somewhere else in the glycan (but the intended motif was still there) (7558d9b)

#### draw
##### Added ✨
- Added the "drawing" argument to `draw_hex`, `hex_circumference`, `add_bond`, `add_sugar`, and `draw_bracket` to avoid having to operate on global variables (918d18f)
- Added the option to provide your own existing glycan .pdb structures to `GlycoDraw` when using `draw_method='chem3d'` with the new keyword argument `pdb_file` (9d082a6)

##### Changed 🔄
- `matches` can now also use `[]` as delimiters (f76535e)
- Support easy import of `GlycoDraw`, via `from glycowork import GlycoDraw` (d5f5d4e)
- Renamed `hex` to `draw_hex`, to avoid overwriting the built-in `hex` (918d18f)
- Changed keyword argument "hex" to "hex_codes" in `add_colours_to_map` (838c708)
- `get_highlight_attribute` now internally uses `motif.graph.subgraph_isomorphism` for pattern retrieval, ensuring up-to-date functionality (4f1ccfa)
- `get_coordinates_and_labels` now internally uses `motif.processing.choose_correct_isoform` to reorder the glycan for drawing (41bb1a1)
- Improved console drawing quality controlled by `display_svg_with_matplotlib` and image quality in Excel cells using `plot_glycans_excel` (a64f694)
- `draw_chem2d` and `draw_chem3d` will now detect whether the user is in a Jupyter environment and, if not, plot to the Matplotlib console (c3a7f64)
- `process_per_residue` now will re-order the `per_residue` list in the same way as the glycan is re-ordered for drawing with `GlycoDraw` (7c510c9)

##### Deprecated ⚠️
- Deprecated `hex_circumference`, the functionality is now available within `draw_hex` with the new keyword argument "outline_only" (4f1ccfa)
- Deprecated `multiple_branches`, `multiple_branch_branches`, `branch_order`, and `reorder_for_drawing` accordingly (41bb1a1)
- Deprecated `matches`, which will now be done by `.processing.get_matching_indices` that has been reworked

##### Fixed 🐛
- Made sure `scale_in_range` never divides by zero, if value range is zero (f76535e)
- Made sure that monosaccharides that were never observed but are still SNFG-defined (like TalNAc vs 6dTalNAc) can still be drawn with `GlycoDraw` (ef24af4)

#### analysis
##### Changed 🔄
- `get_glycanova` will now raise a ValueError if fewer than three groups are provided in the input data (f76535e)
- Improved console drawing quality controlled by `display_svg_with_matplotlib` and image quality in Excel cells using `plot_glycans_excel` (a64f694)
- The "periods" argument in `get_jtk` is now a keyword argument and has a default value of [12, 24] (87ea2fc)
- `specify_linkages` can now also handle super-narrow linkage wildcards like Galb3/4 (f394bda)
- `get_SparCC` will now limit the number of eligible controls for "partial_correlations=True" to sample_size//5, capped at 5 (241141b)

##### Fixed 🐛
- Fixed a FutureWarning in `get_lectin_array` by avoiding DataFrame.groupby with axis=1 (f76535e)
- Fixed a RuntimeWarning in `get_biodiversity` by handling statistical tests of identical alpha diversity values between groups (f76535e)
- Made sure that the TSNE perplexity fits the sample size in `plot_embeddings` (d5f5d4e)
- Fixed an edge case in which user-provided embeddings as DataFrames were misformatted in `plot_embeddings` (d5f5d4e)
- Supported the case where no labels are provided to `plot_embeddings` (d5f5d4e)
- Fixed a potential format mismatch in `get_meta_analysis` if random-effects meta-analyses were performed (d5f5d4e)
- Fixed an issue where variance-filtered rows could cause problems in `get_differential_expression` if "monte_carlo = True" (ef3da9c)
- Fixed an issue in `get_differential_expression` if "sets = True" that caused indexing issues under certain conditions (ef3da9c)
- Ensured that "effect_size_variance = True" in `get_differential_expression` always formats variances correctly (ef3da9c)
- Ensured that the combination of "grouped_BH = True", "paired = False", and CLR/ALR in `get_differential_expression` works even when negative values are present (87ea2fc)

#### regex
##### Changed 🔄
- Improved tracing in `try_matching` for complicated branching cases (f394bda)
- Ensured that `format_retrieved_matches` outputs the identified motifs in the canonical IUPAC representation (7558d9b)

##### Deprecated ⚠️
- Deprecated `process_pattern`; will be done in-line instead (f394bda)
- Deprecated `expand_pattern`; will be handled by `specify_linkages` and improvements in `subgraph_isomorphism` instead (f394bda)
- Deprecated `filter_dealbreakers`; will be handled by improvements in `subgraph_isomorphism` instead (65bd12c)

##### Fixed 🐛
- Fixed an issue in `get_match_batch`, in which precompiled patterns caused issues in `get_match` (194f31c)

#### annotate
##### Added ✨
- Added `get_size_branching_features` to create glycan size and branching level features for downstream analysis (d57b836)
- Added the "size_branch" option in the "feature_set" keyword argument of `annotate_dataset` and `quantify_motifs`, to analyze glycans by size or level of branching (d57b836)

##### Fixed 🐛
- Fixed an issue in `clean_up_heatmap` in which, occasionally, duplicate strings were introduced in the output (e3eeb32)

### ml
#### model_training
##### Added ✨
- Added classification-AUROC, multilabel-accuracy, multilabel-MCC, regression-MAE, and regression-R2 as metrics to `train_model` (#66)
- Added the "return_metrics" keyword argument to `train_model` that can additionally return all training and validation metrics (#66)

##### Changed 🔄
- Weigh metric calculation by batch-size (correctly handling the last batch) in `train_model` (#66)
- Best performances in `train_model` are now taken from the overall best model (lowest loss), not from best-model-per-metric (#66)

##### Fixed 🐛
- Fixed an indexing issue in `train_ml_model` if "additional_features_train" / "additional_features_test" were used (b94744e)

#### inference
##### Changed 🔄
- Changed `resources.open_text` to `resources.files` to prevent `DeprecationWarning` from `importlib` (d1a8c6d)

#### models
##### Changed 🔄
- In `prep_model`, the `hidden_dim` argument can now also be used to modify the protein embedding size of a newly defined LectinOracle model 

### network
#### evolution
##### Fixed 🐛
- Fixed DeprecationWarning in `distance_from_embeddings` to prevent DataFrameGroupBy.apply from operating on the grouping columns (94646ad)
- Fixed an issue in `distance_from_metric` where networks were indexed incorrectly based on presented DataFrame order (d2f5d55)

#### biosynthesis
##### Changed 🔄
- Made sure in `network_alignment` that only nodes that are virtual in all aligned networks stay virtual (918d18f)
- `choose_leaves_to_extend` will now correctly return no leaf node glycan if the target composition cannot be reached from any of the leaf nodes in a network (918d18f)

##### Fixed 🐛
- Fixed an issue in `find_shared_virtuals` in which no shared nodes were found because of graph comparisons (d2f5d55)
