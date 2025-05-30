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