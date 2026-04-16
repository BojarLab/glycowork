# Changelog

## [1.8.1]
- fixed `deploy` action internally still relying on `nbdev2` (84134fb)

### glycan_data
#### loader
##### Added ✨

##### Changed 🔄
- Changed `human_macrophages_N_2024-11-28-625934` and `human_macrophages_O_2024-11-28-625934` glycomics datasets to `human_macrophages_N_2024_11_28_625934` and `human_macrophages_O_2024_11_28_625934`

##### Fixed 🐛
- Fixed one faulty sequence in `df_glycan` that caused graph generation to fail (b2f6ab9)

##### Deprecated ⚠️

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
- Added a `mass_tag` keyword argument to `mz_to_composition` and `mz_to_structures` for glycans tagged at the reducing end

##### Fixed 🐛
- Fixed mass calculation of additionally acetylated glycans in `glycan_to_mass`

### network
#### biosynthesis
##### Added ✨
- Added `get_biosynthetic_coherence` function to estimate how well glycan abundances can be predicted from biosynthetic networks, to disentangle biosynthetic vs carrier variance (b7020fd)