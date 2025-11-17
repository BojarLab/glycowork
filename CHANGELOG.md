# Changelog

## [1.6.5]
- Added some more lazy loading of drawing-related imports to improve package start-up time (b806bdd)

### motif
#### processing
##### Added ✨
- Added the `verbose` keyword argument (default = True) to `glytoucan_to_glycan`, to suppress the output of non-matched IDs (aa0b9a4)
- Added the `kcf_to_iupac` function to convert the KCF nomenclature into IUPAC-condensed (bfe947c, 37fad0d)
- Added the `glycoctxml_to_iupac` function to convert the GlycoCT XML nomenclature into IUPAC-condensed

##### Changed 🔄
- Improved the detection of `LinearCode` sequences in `canonicalize_iupac` with the new `looks_like_linearcode` helper function (e26566e)
- Improved the hook for triggering Oxford nomenclature conversion in `canonicalize_iupac` to be less permissive and faster (c5ae5e5, 25186e5)
- Renamed `linearcode1d_to_iupac` to `glyseeker_to_iupac` (d44aba1)
- Improved the hook for checking GlyTouCan IDs in `canonicalize_iupac` by making it more specific (aa0b9a4)
- Improved `wurcs_to_iupac` handling of complex sequences to have more robust WURCS handling in `canonicalize_iupac` (0f5a5d8)
- Improved `glycoworkbench_to_iupac` handling of variable reducing ends to have more robust GlycoWorkbench handling in `canonicalize_iupac` (0f5a5d8)
- `canonicalize_iupac` can now also convert KCF sequences, thanks to the new `kcf_to_iupac` function (bfe947c, 37fad0d)
- `canonicalize_iupac` can now also convert GlycoCT XML sequences, thanks to the new `glycoctxml_to_iupac` function

##### Fixed 🐛

##### Deprecated ⚠️

#### draw
##### Added ✨
- Added the `reducing_end_label` keyword argument to `GlycoDraw` to display any connected text to the right of the glycan (such as "protein", which will be connected via a regular linkage) (c36a1a6)
- Added the `GlycanDrawing` class to allow `GlycoDraw` to output `glycorender` aesthetics in a Jupyter notebook context

#### annotate
##### Added ✨
- Added the `get_glycan_similarity` function to calculate cosine similarities between glycan motif fingerprints between two glycan sequences (bd4d071)

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