# Changelog

## [1.7.0]

### motif
#### processing
##### Added ✨
- `COMMON_ENANTIOMER` dict to track the implicit enantiomer state (e.g., we write `Gal` instead of `D-Gal` but we do note the deviation `L-Gal`) (bb7575c)

##### Changed 🔄
- `canonicalize_iupac` and its parsers will now leave the `D-/L-` prefixes in monosaccharides, which will then be centrally homogenized with `COMMON_ENANTIOMER`, for a more refined and detailed output (bb7575c)

##### Fixed 🐛
##### Deprecated ⚠️