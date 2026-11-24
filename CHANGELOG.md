# Changelog

## [1.12.1]
- Raised required `viscose` version to `0.3.2` ()

### motif
#### processing
##### Added ✨
- Added `glycan_to_csdb`, which writes CSDB Linear as the Carbohydrate Structure Database does, and `csdb_to_iupac`, which reads it ()

##### Changed 🔄
- `canonicalize_iupac` reads CSDB Linear residue by residue, so diamino sugars (`RhaNAc3NAc`, `ManNAcA3NAc`), uronamides, phosphoethanolamines, lactyl and 3-hydroxybutyryl groups get lib's names ()

##### Fixed 🐛

##### Deprecated ⚠️