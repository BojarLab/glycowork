# Changelog

## [1.7.0]

### motif
#### processing
##### Added ✨

##### Changed 🔄
- Improved the detection of `LinearCode` sequences in `canonicalize_iupac` with the new `looks_like_linearcode` helper function (e26566e)
- Improved the hook for triggering Oxford nomenclature conversion in `canonicalize_iupac` to be less permissive and faster (c5ae5e5, 25186e5)
- Renamed `linearcode1d_to_iupac` to `glyseeker_to_iupac` (d44aba1)

##### Fixed 🐛

##### Deprecated ⚠️

#### data
##### Added ✨
- Added the data processing code for the GlycoGym benchmark collection to create new benchmark releases quickly and to enable local creation of custom GlycoGym-like datasets

#### draw
##### Added ✨
- Added the `reducing_end_label` keyword argument to `GlycoDraw` to display any connected text to the right of the glycan (such as "protein", which will be connected via a regular linkage)

### ml
#### model_training
##### Changed 🔄
- `train_model` now supports training GIFFLAR-type glycan models (1a7e720, 18f42e5)

#### models
##### Added ✨
- The GIFFLAR model can now be requested from `prep_model` via "GIFFLAR" as `model_type` (1a7e720, 18f42e5)