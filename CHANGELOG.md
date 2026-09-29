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
- Added the new `plot_glycans_grid` function to draw glycans with a shared symbol size into a grid (0967d8a)
- `GlycoDraw` now also accepts alternative composition formats (e.g., `{'Hex': 5}`) (0967d8a)
- `GlycoDraw` now accepts lists as input, which are fed into the new `plot_glycans_grid` (0967d8a)
- `plot_glycans_excel` can now pass on styling keyword arguments (e.g., `compact`, `vertical`, etc) to `GlycoDraw` (0967d8a)

##### Changed 🔄
- Black elements in `.svg` `GlycoDraw` outputs are now also charcoal (`#1C1917`), just like any other format (0967d8a)
- Symbols in vertical mode are no longer rotated in `GlycoDraw`, in accordance with SNFG (0967d8a)
- `GlycoDraw` outputs now also carry metadata in notebook outputs as well as Excel cells (0967d8a)
- The `folder_filepath` argument in `plot_glycans_excel` can now also be used to specify a desired output file name (0967d8a)
- Better layouting of complex floating bits in `GlycoDraw` (0967d8a)

##### Fixed 🐛
- Fixed `Altf` and `Tagf` not being recognized as furanoses by `GlycoDraw` (0967d8a)
- Fixed `NS` not being recognized as a joint modification by `GlycoDraw` (0967d8a)
- Fixed `draw_method='chem2d', filepath='x.png'` not saving a `png` (0967d8a)

#### tokenization
##### Changed 🔄
- `get_core` and `get_modification` can now better deal with cases such as `Neu5,9Ac2` (0967d8a)
- `map_to_basic` is now cached, making functions that call it repeatedly (e.g., `structure_to_basic`) about 2x faster (e1c807a)
- `mz_to_composition` and related functions are now faster (eac80ee)

#### processing
##### Added ✨
- The LINUCS nomenclature is now also supported in Universal Input/`canonicalize_iupac` via the new `linucs_to_iupac` parser (e1c807a)
- Universal Input/`canonicalize_composition` now also supports composition nomenclatures from Byonic, FragPipe, GlycoMod, and GlycReSoft (e1c807a)

##### Changed 🔄
- `canonicalize_composition` now also accepts dictionary compositions as inputs (0967d8a)
- `canoncalize_iupac` is more robust to nomenclature variations (e1c807a)

##### Fixed 🐛
- `sanitize_iupac` no longer flags phosphodiesters (e1c807a)
- Fixed handling of variantly capitalized monosaccharides in KCF (e1c807a)

#### annotate
##### Changed 🔄
- Motif annotation is now generally faster (eac80ee)
- Motif annotations with `feature_set` `terminal` are now ~10x faster (eac80ee)

#### analysis
##### Changed 🔄
- `get_glycanova` is now faster (eac80ee)