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

### ml
#### model_training
##### Changed 🔄
- `train_model` now supports training GIFFLAR-type glycan models (1a7e720)

#### models
##### Added ✨
- The GIFFLAR model can now be requested from `prep_model` via "GIFFLAR" as `model_type` (1a7e720)