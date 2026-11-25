# Changelog

## [1.12.1]
- Raised required `viscose` version to `0.3.2` (1b93b15)

### motif
#### processing
##### Added ✨
- Added `glycan_to_csdb`, which writes CSDB Linear as the Carbohydrate Structure Database does, and `csdb_to_iupac`, which reads it (1b93b15)

##### Changed 🔄
- `canonicalize_iupac` reads CSDB Linear residue by residue, so diamino sugars (`RhaNAc3NAc`, `ManNAcA3NAc`), uronamides, phosphoethanolamines, lactyl and 3-hydroxybutyryl groups get lib's names (1b93b15)

##### Fixed 🐛

##### Deprecated ⚠️

#### draw
##### Fixed 🐛
- `GlycoDraw` starts a reducing-end label just past the end of its bond, whatever its width; the label ran into the bond, and in compact mode across the root symbol ()
- SVGs saved from `GlycoDraw` keep `glycorender`'s letter spacing of 0.05 em, so their labels are as wide as in the PNG and PDF ()
- `annotate_figure` widens any SVG to hold the glycans it places to the right of everything, which landed off the page of SVGs not made by `viscose` ()

#### analysis
##### Fixed 🐛
- `get_pca` plots every sample when the metadata has a `sample` column, which left the plot empty ()

### glycan_data
#### data_entry
##### Fixed 🐛
- `read_glycoproteomics(level = 'psm')` keeps GlycanFinder's fraction in the run (`<file>:F1`), so the same scan number in two fractions stays two spectra ()