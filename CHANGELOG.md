# Changelog

## [1.8.1]
- fixed `deploy` action internally still relying on `nbdev2` (84134fb)

### glycan_data
#### loader
##### Added ✨

##### Changed 🔄
- Changed `human_macrophages_N_2024-11-28-625934` and `human_macrophages_O_2024-11-28-625934` glycomics datasets to `human_macrophages_N_2024_11_28_625934` and `human_macrophages_O_2024_11_28_625934` (6a673d0)
- Recurated `human_brain_GSL_PMID40207879` glycomics dataset with improved nomenclature conversion (cf8706a)

##### Fixed 🐛
- Fixed one faulty sequence in `df_glycan` that caused graph generation to fail (b2f6ab9)

##### Deprecated ⚠️

#### stats
##### Added ✨
- Added `hsic` to calculate Hilbert-Schmidt Independence Criterion between variables, to measure dependency (c560fbb)

### motif
#### analysis
##### Changed 🔄
- Added distance matrix to beta diversity output in `get_biodiversity` (dca7820)

##### Fixed 🐛
- Fixed column names slipping into column values when `motifs = True` combined with `transform = ALR` in `get_pca` (e802da1)
- Made motif abundance re-normalization more robust in `preprocess_data` (ac6fa53)

#### draw
##### Changed 🔄
- Improved branch spacing in `GlycoDraw` for highly branched glycans (6a673d0)

#### tokenization
##### Added ✨
- Added a `mass_tag` float keyword argument to `mz_to_composition` and `mz_to_structures` for glycans tagged at the reducing end (7adaf75)
- Added a `modification` string keyword argument to `composition_to_mass` and `glycan_to_mass` for glycans tagged at the reducing end (8160490)
- `mz_to_composition` and `mz_to_structures` now also support the combination of multiply-charged ions with adducts (8160490)

##### Fixed 🐛
- Fixed mass calculation of additionally acetylated glycans in `glycan_to_mass` (7adaf75)

##### Deprecated ⚠️
- The `reduced` bool keyword argument in `mz_to_composition` and `mz_to_structures` has been replaced with the `modification` string keyword argument (8160490)
- Deprecated the `reducing_end` keyword argument in `match_composition_relaxed`, as it was no longer being used

##### Fixed 🐛
- Fixed a bug in `mask_rare_glycoletters` in which rare linkages occasionally were not masked

#### processing
##### Added ✨
- Added some more lipid shorthands (e.g., `Fuc-GD1a` or `Fuc-GA1`) to Universal Input/`canonicalize_iupac` (cf8706a)

##### Changed 🔄
- Universal Input via `canonicalize_iupac` can now deal with more pyranose indicators (e.g., `Altp` or `Lyxp`) (467673b)
- Universal Input via `canonicalize_iupac` can now deal with more sulfate variants (e.g., `6-O-sulfo`, `[S-6]`) (0e102a6, cbe20da)

##### Fixed 🐛
- Fixed overeager modification of already correctly formatted `6PCho` modifications in `canonicalize_iupac` (467673b)
- Fixed `max_specify_glycan` not specifying the chitobiose core in N-glycans

### network
#### biosynthesis
##### Added ✨
- Added `get_biosynthetic_coherence` function to estimate how well glycan abundances can be predicted from biosynthetic networks, to disentangle biosynthetic vs carrier variance (b7020fd)