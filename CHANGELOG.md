# Changelog

## [1.8.2]
- Bumped Python version to `3.11` (`3.10` reaches end of life in October 2026; https://devguide.python.org/versions/) (6a94ea9)
- Bumped minimum `pandas` version to `2.1` (faf68f0)
- Fixed `scipy` version as `>=1.16` to guarantee Games-Howell test in the ANOVA for biodiversity tests (cf5e7bd)

### motif
#### graph
##### Added ✨

##### Changed 🔄
- Made glycan graph caching in `glycan_graph_memoize` somewhat faster (cf5e7bd)

##### Fixed 🐛
- Fixed some degree calculations in `generate_graph_features` (cf5e7bd)

##### Deprecated ⚠️

#### processing
##### Added ✨
- Universal Input via `canonicalize_iupac` can now deal with more cases, such as `Ribp` or `Glc1OMe`

##### Changed 🔄
- Universal Input via `canonicalize_iupac` now is more robust to modified reducing ends in IUPAC-extended glycans