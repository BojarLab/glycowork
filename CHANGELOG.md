# Changelog

## [1.7.0]

### motif
#### draw
##### Added ✨
- `GlycoDraw` now has the new keyword argument `highlight_linkages`, which will draw selected linkages in red and thicker (6b60a53)

##### Changed 🔄

##### Deprecated ⚠️

#### analysis
##### Added ✨
- `get_pca` now has the new keyword argument `size`, to let users control the size of points with a scalar column in the provided meta-data (ab7669c)

#### graph
##### Fixed 🐛
- Fixed an issue in `subgraph_isomorphism_with_negation`, where motif graphs were only shallowly copied, potentially causing graph mutation during processing and leading to too permissive matching in `annotate_dataset` and higher-level functions