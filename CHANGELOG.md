# Changelog

## [1.10.2]

### motif
#### smiles
##### Added ✨
- Added support for phospho-/sulfodiester in SMILES conversion (e5e8631)
- Added support for acyl chain modifications (01253cb)

##### Changed 🔄

##### Fixed 🐛
- Fixed `(2R,3R) L-threo` to `D-erythro (2S,3R)` sphingosine as a default for ceramide (01253cb)
- Fixed amino acid configuration sometimes not defaulting to `L` (01253cb)
- Fixed handling of `1N`-modified monosaccharides such as `GlcNAc1N` and halogen-substituted monosaccharides such as `Gal6F` (01253cb)
- Fixed alditols engaging in glycosidic linkages (01253cb)
- Various other fixes for more exotic modifications (01253cb)

##### Deprecated ⚠️

#### draw
##### Added ✨
- Added the new `plot_glycans_grid` function to draw glycans with a shared symbol size into a grid ()
- `GlycoDraw` now also accepts alternative composition formats (e.g., `{'Hex': 5}`) ()
- `GlycoDraw` now accepts lists as input, which are fed into the new `plot_glycans_grid` ()
- `plot_glycans_excel` can now pass on styling keyword arguments (e.g., `compact`, `vertical`, etc) to `GlycoDraw` ()

##### Changed 🔄
- Black elements in `.svg` `GlycoDraw` outputs are now also charcoal (`#1C1917`), just like any other format ()
- Symbols in vertical mode are no longer rotated in `GlycoDraw`, in accordance with SNFG ()
- `GlycoDraw` outputs now also carry metadata in notebook outputs as well as Excel cells ()
- The `folder_filepath` argument in `plot_glycans_excel` can now also be used to specify a desired output file name ()
- Better layouting of complex floating bits in `GlycoDraw` ()

##### Fixed 🐛
- Fixed `Altf` and `Tagf` not being recognized as furanoses by `GlycoDraw` ()
- Fixed `NS` not being recognized as a joint modification by `GlycoDraw` ()
- Fixed `draw_method='chem2d', filepath='x.png'` not saving a `png` ()

#### tokenization
##### Changed 🔄
- `get_core` and `get_modification` can now better deal with cases such as `Neu5,9Ac2` ()

#### processing
##### Changed 🔄
- `canonicalize_composition` now also accepts dictionary compositions as inputs ()