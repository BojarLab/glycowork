# Changelog

## [1.7.1]

### motif
#### draw
##### Added ✨

##### Changed 🔄
- Generic substituents will now be properly formatted in `GlycoDraw` (89eb687)
- Unknown base monosaccharides in `GlycoDraw` now correctly default to blank hexagons (89eb687)

##### Fixed 🐛
- make sure `reducing_end_label` is perfectly y-centered in `GlycoDraw` (7e9e980)

##### Deprecated ⚠️

#### processing
##### Added ✨
- Added `LacdiNAc` to the `common_names` support in Universal Input (d1140d1)

##### Fixed 🐛
- `canonicalize_iupac` is now more robust when handling variant modification dialects in IUPAC-condensed (i.e., not mistaking them for CSDB-linear), such as `Galβ1-3(6SGlcNAcβ1-6)GalNAcol`

### ml
#### models
- When using `prep_model` with `trained=True` on `SweetNet`-type models, the function now auto-corrects the `num_classes` value, if a wrong output dimension is provided (i.e., if it clashes with the trained model) (ccf2d34)