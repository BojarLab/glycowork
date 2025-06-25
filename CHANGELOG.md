# Changelog

## [1.7.0]
- `glycowork` is now compatible with specifying narrow modification ambiguities (e.g., `Gal(b1-3)GalNAc4/6S`) (ec290e8)
- made the `bokeh` dependency runtime-optional by importing it just-in-time for `plot_network`

### motif
#### processing
##### Added ✨
- `COMMON_ENANTIOMER` dict to track the implicit enantiomer state (e.g., we write `Gal` instead of `D-Gal` but we do note the deviation `L-Gal`) (bb7575c)
- `GLYCONNECT_TO_GLYTOUCAN` dict to support GlyConnect IDs as input to Universal Input / `canonicalize_iupac`

##### Changed 🔄
- `canonicalize_iupac` and its parsers will now leave the `D-/L-` prefixes in monosaccharides, which will then be centrally homogenized with `COMMON_ENANTIOMER`, for a more refined and detailed output (bb7575c)

##### Fixed 🐛
##### Deprecated ⚠️

#### processing
##### Changed 🔄
- `glycan_to_composition` is now compatible with the new narrow modification ambiguities (e.g., `Gal(b1-3)GalNAc4/6S`) (ec290e8)

#### graph
##### Changed 🔄
- `compare_glycans` is now compatible with the new narrow modification ambiguities (e.g., `Gal(b1-3)GalNAc4/6S`) (ec290e8)