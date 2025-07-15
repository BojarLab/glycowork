# Changelog

## [1.6.3]
- `glycowork` is now compatible with specifying narrow modification ambiguities (e.g., `Gal(b1-3)GalNAc4/6S`) (ec290e8)
- made the `bokeh` dependency runtime-optional by importing it just-in-time for `plot_network` (ea9929e)

### motif
#### processing
##### Added ✨
- `COMMON_ENANTIOMER` dict to track the implicit enantiomer state (e.g., we write `Gal` instead of `D-Gal` but we do note the deviation `L-Gal`) (bb7575c)
- `GLYCONNECT_TO_GLYTOUCAN` dict to support GlyConnect IDs as input to Universal Input / `canonicalize_iupac` (ea9929e)

##### Changed 🔄
- `canonicalize_iupac` and its parsers will now leave the `D-/L-` prefixes in monosaccharides, which will then be centrally homogenized with `COMMON_ENANTIOMER`, for a more refined and detailed output (bb7575c)
- `canonicalize_iupac` now considers more IUPAC variations, such as `Neu5,9Ac` instead of `Neu5,9Ac2` (a764897)

##### Deprecated ⚠️

#### processing
##### Changed 🔄
- `glycan_to_composition` is now compatible with the new narrow modification ambiguities (e.g., `Gal(b1-3)GalNAc4/6S`) (ec290e8)
- `wurcs_to_iupac` can now process sulfur linkages (e.g., `Glc(b1-S-4)Glc`) (88b2d54)
- `wurcs_to_iupac` is now more robust to prefixes (e.g., `L-`, `6-deoxy-`, etc)

#### graph
##### Changed 🔄
- `compare_glycans` is now compatible with the new narrow modification ambiguities (e.g., `Gal(b1-3)GalNAc4/6S`) (ec290e8)

#### draw
##### Fixed 🐛
- fixed overlap in floating substituents in `GlycoDraw` if glycan had fewer branching levels than unique floating substituents (daade78)