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
- Universal Input via `canonicalize_iupac` can now deal with more cases, such as `Ribp` or `Glc1OMe` (6754bbf)
- Universal Input via `canonicalize_iupac` can now more robustly handle modifications in CSDB-linear, such as in `Ac(1-5)aXNeup(2-6)[Ac(1-2)]bDGalpN(1-4)bDGalp(1-4)bDGlcp`, `S-3)bDGlcpA(1-3)bDGalp(1-4)[Ac(1-2)]bDGlcpN(1-3)bDGalp(1-4)bDGlcp`, or `Ac(1-2)[xXEt?N(1-P-6)]bDGlcpN(1-3)bDManp(1-4)bDGlcp`, as well as more robustly strip reducing end anomeric indicator (99942e8, a152c47, d6c3d56)

##### Changed 🔄
- Universal Input via `canonicalize_iupac` now is more robust to modified reducing ends in IUPAC-extended glycans (6754bbf)

#### tokenization
##### Changed 🔄
- `mz_to_composition` now also filters by provided `glycan_class` if a user provides a custom `df_use`