# Changelog

## [1.7.0]
- The required version for `glyles`, when using the `[chem]` or `[all]` optional installs, has been bumped up to `1.2.3a0` to resolve dependency conflicts (27eb990)
- Added new glycomics, glycoproteomics, and lectin microarray datasets (5558f1e)
- Added link to `canonicalize` web app into README (3c862db)

### motif
#### draw
##### Added ✨
- `GlycoDraw` now has the new keyword argument `highlight_linkages`, which will draw selected linkages in red and thicker (6b60a53)

##### Deprecated ⚠️

#### analysis
##### Added ✨
- `get_pca` now has the new keyword argument `size`, to let users control the size of points with a scalar column in the provided meta-data (ab7669c)

#### graph
##### Fixed 🐛
- Fixed an issue in `subgraph_isomorphism_with_negation`, where motif graphs were only shallowly copied, potentially causing graph mutation during processing and leading to too permissive matching in `annotate_dataset` and higher-level functions (8b75aae)

#### processing
##### Changed 🔄
- Using `canonicalize_iupac` on a monosaccharide contained in `lib` now has an early return, preventing overlapping name spaces with the common names (040cbc8)
- Added support for old 'z' uncertainty notation in `canonicalize_iupac` via `replace_dic` (b079ece)
- Support C2-inference for beta-linked sialic acid in `canonicalize_iupac`
- Support CarbBank IUPAC dialect in `canonicalize_iupac`