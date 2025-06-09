# Changelog

## [1.6.2]

### glycan_data
#### loader
##### Changed 🔄
- `huggingface_hub` will now only be imported upon running `download_model`, making it technically run-time optional and improving package start-up time (d87e8af)

### motif
#### graph
##### Added ✨

##### Changed 🔄

##### Fixed 🐛
##### Deprecated ⚠️

#### draw
##### Changed 🔄
- `openpyxl` will now only be imported upon running `plot_glycans_excel`, making it technically run-time optional and improving package start-up time (d87e8af)

#### processing
##### Added ✨
- `canonicalize_iupac` now removes extraneous quote marks around input glycans (fbe454c)
- Added more milk oligosaccharide common names to the Universal Input pipeline as recognized by `canonicalize_iupac`

##### Changed 🔄
- `canonicalize_iupac` will now recognize `GLYCAM` sequences terminating in `-OME` (6430ebb)

##### Fixed 🐛
- Fixed capitalisation in mapping of IGG N-glycan codes to account for `.lower()` call in `canonicalize_iupac` (48fb211)
- Fixed variant `LDManHep` handling in `canonicalize_iupac` (6430ebb)