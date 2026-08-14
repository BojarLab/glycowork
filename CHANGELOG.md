# Changelog

## [1.10.0]

- Bumped required `glycorender` version from `0.2.5` to `0.4.0`, which drops the `reportlab` and `pymupdf` dependencies (ec7a54c, 4894d1b, 9741f72)
- Moved `huggingface_hub` dependencies from the base install to the `ml` optional install (088c711)
- `Pillow` is no longer a dependency of `glycowork` (088c711)
- Moved several dependencies to lazy-load, to improve initial package start-up times (088c711)
- Floating bits with uncertain attachment points, such as `{Fuc(a1-3/6)}Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc` can now be optionally further specified as `{Gal(b1-4)[Fuc^(a1-3)]GlcNAc|GlcNAc(b1-4)[Fuc^(a1-6)]GlcNAc}Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc`, which is supported by all graph operations, motif annotation, `get_possible_topologies`, and `GlycoDraw` (4894d1b)

### glycan_data
#### loader
##### Added ✨
- Added the `meta_filter` attribute to `GlycoDataFrame`, to filter datasets by metadata, such as `df_glycan.meta_filter(Order = 'Perissodactyla')`, which can be chained like any `pandas` attribute, such as `df_glycan.meta_filter(Order = 'Perissodactyla').glyco_filter('Sia(a2-3)Gal')` (c11b4d0)
- Added newly curated comparative glycomics datasets to `glycomics_data_loader`: `human_serum_parkinson_GSL_PMID40379659`, `mouse_brain_tango2ko_GSL_10_1002pgr2_70042`, `mouse_brain_tango2ko_N_10_1002pgr2_70042`, `mouse_brain_tango2ko_O_10_1002pgr2_70042`, `fish_gill_infection_O_PMID41435595`, `fish_intestine_infection_O_10_2139ssrn_7005880`, `human_colorectal_butyrate_O_PMID36669592`, `human_celllines_N_PMID38022636`, `human_serum_gangliosidosis_GSL_PMID39190143` (dfbc225)

##### Changed 🔄
- Changed the `GlycoDataFrame` attribute `_name` to `_glyco_name` to avoid shadowing the `pandas` attribute (9afa2b3)

#### stats
##### Added ✨
- `clr_transformation` now has a new `reference` keyword argument, to optionally specify from which variables the geometric mean should be constructed (a918a2e)
- `MissForest` now runs left-censored draws where missingness is intensity-dependent (MNAR) and Random Forest for the rest (MAR), where intensity-dependence is estimated via logistic regression (7849b15)
- `get_alphaN` now has a new `verbose` keyword argument to suppress its print (which is now default `False`, since its output will now be stored as dataframe attributes) (5483f3c)
- `impute_and_normalize` and `MissForest` now have the new `random_state` keyword argument to make imputation fully reproducible (9630cd0)

##### Changed 🔄
- `hotellings_t2` is now more robust to tiny groups (a918a2e)
- `TST_grouped_benjamini_hochberg` is now more robust to groups with π0 = 1 (c902297)
- `replace_outliers_winsorization` now takes in the entire `df` as input, instead of only one `row` (5483f3c)
- `replace_outliers_winsorization`, `omega_squared`, and `permanova_with_permutation` are now much faster (5483f3c, 0b0085c)
- `mahalanobis_distance` can now properly account for paired data as well (f5ba41f)

##### Fixed 🐛
- Fixed sum of squares calculation in `calculate_permanova_stat` (5483f3c)
- `hsic` is now correctly symmetrical (7bf463b)

### motif
#### tokenization
##### Added ✨
- Added more element masses to `calculate_adduct_mass` (S, P, Na, K, Cl) (8f799c3)

##### Changed 🔄
- `mz_to_composition` now returns the closest prioritized match instead of the first prioritized match within tolerance (28769f5)

##### Fixed 🐛
- Fixed `glycan_to_composition` handling of narrow monosaccharide modification wildcards such as `HexNAc4/6S` (b3ad039)
- `stemify_dataset` no longer mutates its `stem_lib` input (8f799c3)
- Fixed edge cases where `structure_to_basic` could poison the graph cache (78fe195)
- Made sure `pad_sequence` does not mutate its input (78fe195)
- `Na+` and `K+` masses in `mz_to_composition` now correctly use the cationic mass, rather than the neutral mass (1c7a6c0)
- Fixed charge modulator from hydrogen mass to proton mass in `mz_to_composition` (1c7a6c0)

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
- `preprocess_data` now has the new optional keyword argument `glycoproteomics`, mainly for `get_differential_expression` to pass this info to be able to trigger `get_composition_dag` (c11b4d0)
- `get_glycanova`, `get_time_series`, and `get_jtk` now have the new optional keyword argument `glycoproteomics`, to facilitate ANOVA-type, time series, and circadian glycoform analysis of glycoproteomics data (c11b4d0, f3ee7e0)
- Added the new optional keyword argument `random_state` to `get_time_series`, `get_jtk`, and `get_SparCC` to make them fully reproducible (9630cd0)
- Added the new optional keyword argument `moderate_variance` to `get_differential_expression` and `get_glycanova` to support Empirical-Bayes variance moderation (afa0634)

##### Changed 🔄
- `get_pca` now correctly filters out redundant motifs for the PCA analysis (a918a2e)
- `get_differential_expression`, `get_glycanova`, `get_time_series`, and `get_jtk` now default to `grouped_BH=True` if they are run in motif-analysis mode (`motifs=True`) (7849b15)
- `preprocess_data` now propagates `random_state` to data imputation for full re-run reprodubility (9630cd0)
- The p-value in the polynomial case in `get_glycan_change_over_time` now has a better relationship with the trend line (rather than a mean shift) (f5ba41f)
- `get_pvals_motifs` has been brought in line with the other analysis functions: enrichment is now tested with an Empirical-Bayes moderated t-test using the containment DAG as variance prior (new `moderate_variance` keyword argument), corrected by two-stage Benjamini-Hochberg within DAG-grouped motif families (new `grouped_BH` keyword argument) against a sample-size-adjusted alpha, on a motif set deduplicated via `deduplicate_motifs`, and additionally reports `significant` and `equivalence_pval` columns (42edfc9)
- `get_representative_substructures` now uses the sample-size-adjusted significance from `get_pvals_motifs`, instead of a hardcoded corrected p-value threshold of 0.05 (42edfc9)

##### Fixed 🐛
- `get_time_series` and `get_jtk` now correctly do motif quantification followed by CLR/ALR (instead of the other way around) in case of motif-analysis (a918a2e)
- `get_pvals_motifs` no longer mutates input dataframe (8f799c3)
- Fixed effect size handling in `get_meta_analysis` if `model="random"` (8f799c3)
- Made sure `get_SparCC` does not mutate its inputs (78fe195)
- `get_representative_substructures` no longer crashes if only run on motif outputs of `feature_set=["known"]` (78fe195)
- Fixed `get_pvals_motifs` calculating effect sizes from padded arrays, which gave motifs that do not occur in the data a large spurious Cohen's d and, given the default sorting, placed them at the top of the output (42edfc9)
- Fixed `get_pvals_motifs` assuming the glycan column is the first column when z-scoring and renaming, rather than the detected glycan column (42edfc9)

#### processing
##### Changed 🔄
- `de_wildcard_glycoletter` now also supports narrow monosaccharide wildcards like `Gal/Glc` (5483f3c)
- `canonicalize_iupac` is again made more robust in terms of what inputs it can handle (8f799c3)
- `UND` tokens in `GlycoCT` inputs are now better supported in `canonicalize_iupac` (4894d1b)

##### Fixed 🐛
- Fixed `canonicalize_iupac` messing up narrow modification wildcards (e.g., `Gal3/6S`) in side branches (3a02eff)
- Fixed `canonicalize_iupac` sometimes ordering multiple floating bits wrongly (b6689a5)
- Fixed `canonicalize_iupac` crashing on empty string inputs (8f799c3)
- `de_wildcard_glycoletter` now no longer can draw and return the wildcard itself (8f799c3)
- Made sure `process_for_glycoshift` does not mutate its input (78fe195)

#### draw
##### Added ✨
- The `highlight_motif` argument in `GlycoDraw` now also accepts motif common names (such as `Internal_LewisX`) (8925497)
- `GlycoDraw` has a new optional `shadow=False` keyword argument, to give the SNFG monosaccharides a drop shadow, if desired (9741f72)

##### Changed 🔄
- Drawing "forbidden" monosaccharides, such as in `GlycoDraw("Gal(b1-3)[!GlcNAc(b1-6)]GalNAc")`, now automatically makes the forbidden monosaccharides and their linkages transparent (3621350)
- Saved `GlycoDraw` outputs are now cropped much more tightly, producing less whitespace around the glycan (f3ee7e0)
- Using named motifs in `GlycoDraw`, such as `Internal_LewisX` is now robust to variant capitalization, spaces, underscores, and hyphens (8925497)
- `GlycoDraw` drawings with gradients now have a 2x smaller filesize and smoother gradients (9741f72)
- Saved `.png` outputs from `GlycoDraw` now have a transparent background (9741f72)
- Glycan drawings in `annotate_figure` are now positioned much better to reduce overlap (9741f72)
- Improved overlaps/positioning in `GlycoDraw`
- `GlycoDraw` will now automatically reduce the font size of overly long narrow wildcard linkages (e.g., `b1-2/4/6`) to let them fit on linkages

##### Fixed 🐛
- Fixed `GlycoDraw` being unusable if `Jupyter` was not installed (28769f5)
- `annotate_figure` is now much more robust to detect glycan strings in figures and draw them (9741f72)

##### Deprecated ⚠️
- Removed the `show_linkage` keyword argument in `get_coordinates_and_labels` (it was dead and handled within `GlycoDraw`) (f5ba41f)

#### graph
##### Changed 🔄
- `subgraph_isomorphism` and `compare_glycans` have been made more performant (5483f3c)

##### Fixed 🐛
- Fixed `subgraph_isomorphism` counting too many isomorphic matches in certain scenarios of linkage ambiguity (81244c9)
- `get_linkage_number` now always handles narrow linkage wildcards correctly for branch sorting purposes (d222f41)
- Fixed edge case handling of sulfation wildcards in `subgraph_isomorphism` (8f799c3)
- Fixed returned node numbering if glycans returned from the fast `compare_glycans` branch (28769f5)
- Fixed `compare_glycans`/`subgraph_isomorphism` not treating `HexOP`/`HexN` as proper wildcards (0602f71)
- Made sure `get_possible_topologies` doesn't swallow multiple floaty bits past the first one (78fe195)

#### annotate
##### Added ✨
- Added `get_motif_dag` and `get_composition_dag` to build a containment DAG of connected motifs/compositions, for comparative glycomics/glycoproteomics analyses in `analysis` (a918a2e, c11b4d0)

##### Changed 🔄
- Refined counting of `Terminal_` motifs in `annotate_dataset` (a918a2e)
- `deduplicate_motifs` will now (given the choice) always prefer the specified motif over the unspecified motif, all else being equal (e.g., `Fuc` > `dHex`) (81e769c)
- `annotate_dataset` will no longer split signal between pairs such as `Gal(b1-4)GlcNAc` and `Gal(b1-4)GlcNAc-ol` in free oligosaccharides (3737633)

##### Fixed 🐛
- Fixed a row dropping bug if `deduplicate_motifs` was run on non-imputed data (8f799c3)
- Hardened `annotate_dataset` against duplicate motifs if the `"custom"` motif set is used (28769f5)
- Made `annotate_dataset` more robust to glyco-regex custom motifs in `feature_set` (78fe195)

#### regex
##### Fixed 🐛
- Harden treatment of ?-wildcards in `parse_pattern` (8f799c3)
- `filter_matches_by_location` no longer crashes on empty inner matches (9630cd0)

##### Deprecated ⚠️
- Deprecated `all_combinations` (will be handled in-line instead) (9630cd0)

### network
#### biosynthesis
##### Changed 🔄
- In `construct_network`, `edge_type=enzyme` will now assign glycan class-specific enzymes, if possible (e.g., only ST3GAL4 for N-glycans instead of all ST3GALs) (cb262b3)
- Improved graph caching, which should result in faster `construct_network` calls (9630cd0)

#### evolution
##### Changed 🔄
- Milk networks are now loaded lazily instead of eagerly upon module load, improving package start-up time (0b0085c)

##### Fixed 🐛
- Made sure `check_conservation` no longer crashes if a rank has no matching network (78fe195)

### ml
#### model_training
##### Fixed 🐛
- Fixed double softmax in `train_model` (5483f3c)

#### inference
##### Changed 🔄
- `get_esmc_representations` now auto-cleans protein sequences via the new `_clean_protein_sequences` function (8b0a73e)