# Changelog

## [1.8.2]
- Bumped Python version to `3.11` (`3.10` reaches end of life in October 2026; https://devguide.python.org/versions/) (6a94ea9)
- Bumped minimum `pandas` version to `2.1` (faf68f0)
- Fixed `scipy` version as `>=1.16` to guarantee Games-Howell test in the ANOVA for biodiversity tests (cf5e7bd)
- The `dev` optional install set (only used for testing) now also includes `glycontact>=0.3.3`

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
- `mz_to_composition` now also filters by provided `glycan_class` if a user provides a custom `df_use` (ab57479)

##### Fixed 🐛
- `composition_to_mass` now correctly factors in the extra methylation (former ring oxygen) that happens in the combination of `modification == 'reduced'` and `sample_prep == 'permethylated'`

#### annotate
##### Changed 🔄
- Arguments `glycans` and `feature_set` in `quantify_motifs` have been changed to keyword arguments with defaults `glycans = None` (will be inferred from first column if it contains glycans, otherwise needs to be supplied) and `feature_set = ['known', 'exhaustive']` (c5db6a8)

##### Fixed 🐛
- `quantify_motifs` can now also be used with full datasets that still have the first column be a glycan string column (fa98caa)

#### draw
##### Changed 🔄
- If `draw_method = chem3d`, `GlycoDraw` will now preferentially fetch a realistic conformer from GlycoShape/PDB via `glycontact`, if the user has `glycontact` installed (lazily imported), and only fall back to RDKit if none can be found (7a59d08)