# Changelog

## [1.6.1]
- Moved `xgboost` dependency into the optional `[ml]` install (0c62acf)
- `glycowork` now no longer has a `svglib` dependency, due to improvements in `glycorender`, requiring `glycorender[png]>=0.2.0` (4cad68f)

### motif
#### graph
##### Added ✨

##### Changed 🔄
- `glycan_to_nxGraph_int` will now automatically convert provided `lib` dicts into `HashableDict` objects, if they aren't already (fe5cd74)
- `compare_glycans` used with two strings now has another early-return condition if the two glycans have different numbers of branches, enhancing efficiency (fe5cd74)

##### Fixed 🐛
##### Deprecated ⚠️

#### processing
##### Changed 🔄
- `canonicalize_iupac` can now handle some more variations, such as double-anomeric linkages (`(a2-1b)`), and will leave modification-containing-seeming monosaccharides (e.g., `Psif`, `Sorf`) intact