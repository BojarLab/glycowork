# Changelog

## [1.10.2]

### motif
#### smiles
##### Added ✨
- Added support for phospho-/sulfodiester in SMILES conversion (e5e8631)
- Added support for acyl chain modifications ()

##### Changed 🔄

##### Fixed 🐛
- Fixed `(2R,3R) L-threo` to `D-erythro (2S,3R)` sphingosine as a default for ceramide ()
- Fixed amino acid configuration sometimes not defaulting to `L` ()
- Fixed handling of `1N`-modified monosaccharides such as `GlcNAc1N` and halogen-substituted monosaccharides such as `Gal6F` ()
- Fixed alditols engaging in glycosidic linkages ()
- Various other fixes for more exotic modifications ()

##### Deprecated ⚠️
