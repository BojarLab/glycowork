# Changelog

## [1.7.0]
- Moved `xgboost` dependency into the optional `[ml]` install

### motif
#### graph
##### Added ✨

##### Changed 🔄
- `glycan_to_nxGraph_int` will now automatically convert provided `lib` dicts into `HashableDict` objects, if they aren't already (fe5cd74)
- `compare_glycans` used with two strings now has another early-return condition if the two glycans have different numbers of branches, enhancing efficiency (fe5cd74)

##### Fixed 🐛
##### Deprecated ⚠️