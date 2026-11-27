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
##### Added ✨
- Added `draw_glycans_in_sheet`, which draws glycans as SNFG pictures into any openpyxl worksheet, one per row at one scale (compositions too), so a workbook of several sheets can hold them, and `add_svg_twins`, which gives the pictures of a saved workbook their vector twins on every sheet; `plot_glycans_excel` is built on them, its output unchanged ()

##### Fixed 🐛
- `GlycoDraw` keeps an extended bisecting GlcNAc (such as `Gal(b1-4)GlcNAc(b1-4)` on the core Man) in line with the core, between the a1-6 arm above and the a1-3 arm below; it climbed over the a1-6 arm and pushed that arm into the middle ()
- `GlycoDraw` starts a reducing-end label just past the end of its bond, whatever its width; the label ran into the bond, and in compact mode across the root symbol (e78b66e)
- SVGs saved from `GlycoDraw` keep `glycorender`'s letter spacing of 0.05 em, so their labels are as wide as in the PNG and PDF (e78b66e)
- `annotate_figure` widens any SVG to hold the glycans it places to the right of everything, which landed off the page of SVGs not made by `viscose` (e78b66e)
- Fixed layout-ing of extended bisecting GlcNAc structures in `GlycoDraw` ()

#### analysis
##### Fixed 🐛
- `get_pca` plots every sample when the metadata has a `sample` column, which left the plot empty (e78b66e)

### glycan_data
#### data_entry
##### Fixed 🐛
- `read_glycoproteomics(level = 'psm')` keeps GlycanFinder's fraction in the run (`<file>:F1`), so the same scan number in two fractions stays two spectra (e78b66e)