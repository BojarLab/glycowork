# Changelog

## [1.7.0]

### glycan_data
#### loader
##### Changed 🔄
- `huggingface_hub` will now only be imported upon running `download_model`, making it technically run-time optional and improving package start-up time

### motif
#### graph
##### Added ✨

##### Changed 🔄

##### Fixed 🐛
##### Deprecated ⚠️

#### draw
##### Changed 🔄
- `openpyxl` will now only be imported upon running `plot_glycans_excel`, making it technically run-time optional and improving package start-up time

#### processing
##### Added ✨
- `canonicalize_iupac` now removes extraneous quote marks around input glycans

##### Fixed 🐛
- Fixed capitalisation in mapping of IGG N-glycan codes to account for `.lower()` call in `canonicalize_iupac`