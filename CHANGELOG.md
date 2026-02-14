# Changelog

## [1.7.1]
- `glycorender` version bump from `0.2.3` to `0.2.5`  (1933574)
- upgraded `nbdev2` to `nbdev3` for the documentation (+ removed now unnecessary files) (eb3f727)
- improved start-up time of the package (i.e., time at first import in a session)

### motif
#### draw
##### Changed 🔄
- Generic substituents will now be properly formatted in `GlycoDraw` (89eb687)
- Unknown base monosaccharides in `GlycoDraw` now correctly default to blank hexagons (89eb687)
- Make sure `GlycoDraw` can draw !-containing sequences (e.g., `Internal_LewisA`) even with `restrict_vocab=True` (1933574)

##### Fixed 🐛
- Make sure `reducing_end_label` is perfectly y-centered in `GlycoDraw` (7e9e980)
- Fixed setting utf-8 as default encoding in `annotate_figure` (1933574)

##### Deprecated ⚠️

#### processing
##### Added ✨
- Added `LacdiNAc` to the `common_names` support in Universal Input (d1140d1)
- Added `max_specify_glycan` function to infer sequence ambiguities/uncertainties as best as possible (e2cf92a)

##### Fixed 🐛
- `canonicalize_iupac` is now more robust when handling variant modification dialects in IUPAC-condensed (i.e., not mistaking them for CSDB-linear), such as `Galβ1-3(6SGlcNAcβ1-6)GalNAcol` (046ea12)
- `min_process_glycans` and `get_lib` now correctly handle glycans with floating modifications, such as `{6S}{Neu5Ac(a2-3)}Gal(b1-4)GlcNAc(b1-6)[Gal(b1-3)]GalNAc` (68f1e1b)

#### analysis
##### Changed 🔄
- `characterize_monosaccharide` is now much faster (0de71c5)

##### Fixed 🐛
- Fixed temporary file handling in `annotate_volcano=True` in `get_volcano` (1933574)

#### annotate
##### Added ✨
- Added new `get_minimal_ksaccharide_ambiguity` function to find the minimal needed narrow linkage wildcard to encompass all variants in dataset (8a0bbce)

##### Changed 🔄
- `feature_set` options `exhaustive` and the `terminal` variants now fully lean into narrow linkage wildcards for dynamically generated wildcards (e.g., `a2-3/6`), instead of the broader `a2-?` versions, which are scoped based on the provided data (8a0bbce)
- `get_terminal_structures` can now be used for any `size` value, not only 1 and 2 (ef353fb)
- `annotate_dataset` will now internally use `get_terminal_structures` for the `terminal3` feature-set keyword (ef353fb)

##### Fixed 🐛
- Fixed topologically incorrect disaccharides in `get_terminal_structures` output (ef353fb)

### ml
#### models
- When using `prep_model` with `trained=True` on `SweetNet`-type models, the function now auto-corrects the `num_classes` value, if a wrong output dimension is provided (i.e., if it clashes with the trained model) (ccf2d34)
##### Fixed 🐛
- Fixed warning message in `train_ml_model` about not specifying `feature_calc` (0de71c5)