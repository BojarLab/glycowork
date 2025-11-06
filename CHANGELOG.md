# Changelog

## [1.7.0]
- Added some more lazy loading of drawing-related imports to improve package start-up time (b806bdd)

### motif
#### processing
##### Added ✨

##### Changed 🔄
- Improved the detection of `LinearCode` sequences in `canonicalize_iupac` with the new `looks_like_linearcode` helper function (e26566e)
- Improved the hook for triggering Oxford nomenclature conversion in `canonicalize_iupac` to be less permissive and faster (c5ae5e5, 25186e5)
- Renamed `linearcode1d_to_iupac` to `glyseeker_to_iupac` (d44aba1)

##### Fixed 🐛

##### Deprecated ⚠️

#### draw
##### Added ✨
- Added the `reducing_end_label` keyword argument to `GlycoDraw` to display any connected text to the right of the glycan (such as "protein", which will be connected via a regular linkage) (c36a1a6)
- Added the `GlycanDrawing` class to allow `GlycoDraw` to output `glycorender` aesthetics in a Jupyter notebook context

### ml
#### model_training
##### Added ✨
- Added `WarmupScheduler` class to (by default) have a warming-up period of learning rate schedule (for training stability) in `training_setup` (469649a)

##### Changed 🔄
- `train_model` now supports training GIFFLAR-type glycan models (1a7e720, 18f42e5)
- `training_setup` has the new `warmup_epochs` keyword argument (default = 5) that determines the length of the learning rate warm-up schedule (469649a)
- `train_model` now performs gradient clipping for improved training stability (469649a)

#### models
##### Added ✨
- The GIFFLAR model can now be requested from `prep_model` via "GIFFLAR" as `model_type` (1a7e720, 18f42e5)

#### inference
##### Changed 🔄
- Added the `multilabel` keyword argument to `glycans_to_emb`, to support inference for multilabel outputs (0437583)

### glycan_data
#### loader
##### Added ✨
- Added `parse_lines` utility function to parse copy-pasted content from an Excel column into a list (b806bdd)