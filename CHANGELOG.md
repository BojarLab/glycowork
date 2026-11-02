# Changelog

## [1.11.2]

### motif
#### tokenization
##### Added ✨

##### Changed 🔄
- `glycan_to_mass` and `get_formula` now raise a `ValueError` when modification is set but the glycan has no free reducing end (alditols other than reduction, aldonic acids, inositol, aglycones, sucrose/trehalose-type linkages) ()

##### Fixed 🐛

##### Deprecated ⚠️
