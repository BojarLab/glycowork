# Changelog

## [1.9.0]
- Bumped Python version to `3.11` (`3.10` reaches end of life in October 2026; https://devguide.python.org/versions/) (6a94ea9)
- Bumped minimum `pandas` version to `2.1` (faf68f0)
- Fixed `scipy` version as `>=1.16` to guarantee Games-Howell test in the ANOVA for biodiversity tests (cf5e7bd)
- The `dev` optional install set (only used for testing) now also includes `glycontact>=0.3.3`
- Automated versioning in `__init__.py` (b72d113)

### glycan_data
- Curated datasets are now stored in a dedicated folder `.glycan_data.datasets`, for tidiness (c91ee8d)
- Added new curated glycomics datasets: `human_interstitialfluid_N_GPST000651`, `celllines_colorectal_O_PMID32236654`, `celllines_colorectal_GSL_PMID35489554`, `human_serum_ovarian_N_PMID41366864`, `mouse_platelet_N_101161ATVBAHA126324757` (ce499ae, 7ddd6c2, c91ee8d, 086d589, c29ceeb, e6f59e6)
- Renamed and re-curated `glycomics_mouse_gastric_O_GPST000464` to `glycomics_mouse_gastric_O_PMID40667878` (c91ee8d)
- Added new `contrasts` file to catalog which sample belongs to which comparison group for the curated glycomics datasets (aad159e)
- `df_species` now contains a new `taxon_id` column with TaxIDs (3d304a3)

#### loader
##### Added ✨
- Added the `GlycoList` class that equips `List` with graph-based matching capabilities (overriding `index`, `remove`, `count`, and `in` with `compare_glycans` isomorphisms) (49c9aeb)
- Added the `glycans`, `abundance`, `group1`, `group2`, `groups`, `paired`, and `name` properties to `GlycoDataFrame` to benefit from `GlycoList` capabilities in `glycans`, get the abundance values via `abundance` and provide functions in `glycowork.motif.analysis` with group labels and information whether data are paired or not (aad159e, c91ee8d, cd1b9aa)

#### stats
##### Added ✨
- `MissForest` now has a `circadian` API (also exposed in `impute_and_normalize`, to fine-tune data imputation of rhythmic data (starting from the same-phase median, instead of the overall median) (4305377)

##### Fixed 🐛
- Fixed behavior of `hotellings_t2` when `paired=True` (e419fc4)
- Fixed `get_glycoform_diff` mutating its input (e419fc4)

### motif
#### graph
##### Added ✨
- All graph functions (e.g., `compare_glycans`, `subgraph_isomorphism`) now support narrow monosaccharide wildcards (e.g., `Gal/Man` instead of `Hex`) (f85ab80)

##### Changed 🔄
- Made glycan graph caching in `glycan_graph_memoize` somewhat faster (cf5e7bd)

##### Fixed 🐛
- Fixed some degree calculations in `generate_graph_features` (cf5e7bd)
- Fixed `graph_to_string` very rarely not ordering branches in a deterministic/idempotent manner (9f42561)

##### Deprecated ⚠️

#### processing
##### Added ✨
- Universal Input via `canonicalize_iupac` can now deal with more monosaccharide cases, such as `Ribp`, `Glc1OMe`, or single-monosaccharide glycans such as `aDGlcpA` (6754bbf, 426ff9d)
- Universal Input via `canonicalize_iupac` can now more robustly handle modifications in CSDB-linear, such as in `Ac(1-5)aXNeup(2-6)[Ac(1-2)]bDGalpN(1-4)bDGalp(1-4)bDGlcp`, `S-3)bDGlcpA(1-3)bDGalp(1-4)[Ac(1-2)]bDGlcpN(1-3)bDGalp(1-4)bDGlcp`, `-4)[S-2)]aD3,6anhGalp(1-3)[S-2)]bDGalp(1-`, or `Ac(1-2)[xXEt?N(1-P-6)]bDGlcpN(1-3)bDManp(1-4)bDGlcp`, as well as more robustly strip reducing end anomeric indicator (99942e8, a152c47, d6c3d56, ced60fc, a6882db)
- Universal Input via `canonicalize_iupac` can now parse more complex Oxford sequences, such as `F(6)A2G(4)2S(3,3)2` (9f42561)
- `max_specify_glycan` will now also specify these cases: `("Fuc(a1-?)GlcNAc", "Fuc(a1-3/4)GlcNAc")`, `("Fuc(a1-?)]GlcNAc", "Fuc(a1-3/4)]GlcNAc")`, `("Fuc(a1-?)Gal(", "Fuc(a1-2)Gal(")`, `("GalOS", "Gal3/6S")`, `("GlcNAcOS", "GlcNAc6S")`, `("Fuc(a1-?)[Gal(b1-?)]", "Fuc(a1-3/4)[Gal(b1-3/4)]")`, `("Man(a1-?)Man", "Man(a1-2/3/6)Man")` (f3b4389, 487dad4)
- Oxford parsing in `canonicalize_iupac` will now detect hybrid glycans and will add extra mannoses to the `a1-6` branch (c4ba7f9)

##### Changed 🔄
- Universal Input via `canonicalize_iupac` now is more robust to modified reducing ends in IUPAC-extended glycans (6754bbf)
- `canonicalize_iupac` has again been made more robust against typos and personal idiosyncrasies in nomenclature (39b1d25, 9f57fa5, e419fc4)

##### Fixed 🐛
- Fixed some Oxford sequences (e.g., `A1`, `A2`) being misidentified as blood group glycolipids by `canonicalize_iupac` (9f42561)
- Fixed `canonicalize_composition` mistaking sialic acid composition blocks as sulfate in compositions such as `H2N2S1Sulf1` (bb822c8)
- Fixed `oxford_to_iupac` wrongly ordering `NeuOAc5Ac` instead of `Neu5AcOAc` (eda4474)

#### tokenization
##### Added ✨
- The `modification` keyword argument in `mz_to_composition` etc now also accepts `procainamide` as an argument (e4a2a40)
- `mz_to_composition` and `mz_to_structures` now have a new `max_charge` keyword argument that sets the maximum applicable charge state as well as the ion mode (e1475f9)
- `mz_to_composition` and `mz_to_structures` now also support ppm-level mass tolerances via the new `tolerance_unit` keyword argument that allows users to switch between Da and ppm (e1475f9)

##### Changed 🔄
- `mz_to_composition` now also filters by provided `glycan_class` if a user provides a custom `df_use` (ab57479)

##### Fixed 🐛
- `composition_to_mass` now correctly factors in the extra methylation (former ring oxygen) that happens in the combination of `modification == 'reduced'` and `sample_prep == 'permethylated'` (30e1a46)
- Fixed incompatibility of `condense_composition_matching` with `scikit-learn >= 1.9.0` (34d839f)

##### Deprecated ⚠️
- Removed the keyword argument `mode` from `mz_to_composition` and `mz_to_structures`; will be handled by the new `max_charge` instead (e1475f9)
- Removed the `doubly_charged` option from the `extras` keyword argument in `mz_to_composition`; will be handled by the new `max_charge` instead (e1475f9)

#### annotate
##### Added ✨
- `annotate_glycan` now also exposes the keyword argument `condense=False`, analogous to `annotate_dataset`, to get only non-zero motifs (5d4fa70)

##### Changed 🔄
- Arguments `glycans` and `feature_set` in `quantify_motifs` have been changed to keyword arguments with defaults `glycans = None` (will be inferred from first column if it contains glycans, otherwise needs to be supplied) and `feature_set = ['known', 'exhaustive']` (c5db6a8)
- Wildcard monosaccharide-only columns (e.g., `dHex`) in `get_k_saccharide` will now also aggregate signal from their specific instances (e.g., `Fuc`) across the dataset (was already working like that for `Sia` and for k>1) (6de40f9)

##### Fixed 🐛
- `quantify_motifs` can now also be used with full datasets that still have the first column be a glycan string column (fa98caa)
- `get_k_saccharides` will no longer mistakenly capture narrow linkage wildcards like `b1-3/4` in its feature columns for k=1 (where there should only be monosaccharides) (c209f85)

#### draw
##### Added ✨
- When supplied with narrow monosaccharide wildcards, `GlycoDraw` will now draw bisected monosaccharides (e.g., a blue-yellow split rectangle for `GlcNAc/GalNAc`) (f85ab80)

##### Changed 🔄
- If `draw_method = chem3d`, `GlycoDraw` will now preferentially fetch a realistic conformer from GlycoShape/PDB via `glycontact`, if the user has `glycontact` installed (lazily imported), and only fall back to RDKit if none can be found (7a59d08)

##### Fixed 🐛
- Made SVG parsing in `annotate_figure` more robust (e3dce0e)
- Glycans with `/` in their sequence can now be saved with the glycan name as the filename in `GlycoDraw` without error (edba778)

#### analysis
##### Changed 🔄
- Several functions are now more robust toward the specific glycan column naming for processing (aad159e)
- `get_jtk` now internally uses fine-tuned data imputation for rhythmic data via `MissForest` improvements (4305377)

##### Fixed 🐛
- Fixed column access in ALR-treatment of `get_glycanova` (aad159e)
- Fixed warnings when using `scikit-learn>=1.9.0` due to the deprecated `penalty` keyword arg in `sklearn.linear_model.LogisticRegression` used in `multi_feature_scoring` (34d839f)
- Fixed edge case where `get_glycanova` could assign effect sizes to wrong glycans, if glycans had been dropped due to variance filtering (e419fc4)

##### Deprecated ⚠️
- Deprecated `glycan_col_name` keyword argument in `get_pvals_motif` and `characterize_monosaccharide`; will be auto-detected (aad159e)

#### regex
##### Fixed 🐛
- Specifying exact occurrences in `get_match`, as in `get_match("[HexNAc]{2}", "Gal(b1-4)GlcNAc(b1-4)GlcNAc")`, is now more robust/accurate (28894d8)

##### Deprecated ⚠️
- Deprecated `process_occurrence`, `process_main_branch`, and `process_question_mark` as they are handled in-line now (28894d8)
- Deprecated the `lookahead` keyword argument in `fill_missing_in_list` (handled automatically) (487dad4)