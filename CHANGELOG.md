# Changelog

## [1.7.2]
- fixed `Quarto` accessing of `pyproject.toml` attributes for doc building (cd9b62f)

### glycan_data
#### loader
##### Added ✨
- Added new N- and O-glycomics dataset from https://pubmed.ncbi.nlm.nih.gov/41460292/ to `glycomics_data_loader` (`mouse_taysachs_N_PMID41460292` and `mouse_taysachs_O_PMID41460292`)

##### Changed 🔄

##### Fixed 🐛
- Made sure that incomplete API access in `get_molecular_properties` does not lead to outright failure

##### Deprecated ⚠️
