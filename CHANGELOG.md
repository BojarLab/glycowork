# Changelog

## [1.6.0]

## glycan_data
##### Changed 🔄
- refined motif definition of `Internal_LewisX`/`Internal_Lewis_A`/`i_antigen` in `motif_list`, to exclude `LewisY`/`LewisB`/`I_antigen` from matching/overlapping
- renamed `Hyluronan` in `motif_list` into `Hyaluronan`
- removed `Nglycolyl_GM2` from `motif_list`; it's captured by `GM2`

### motif
#### processing
##### Added ✨
- GlyTouCanIDs are now another supported nomenclature in the context of Universal Input and can be used as inputs for functions etc, supported via improvements in `canonicalize_iupac` (eafb218)
- Added `sanitize_iupac` to detect and fix chemical impossibilities (like two monosaccharides connected via the same hydroxyl group) and fix it (407cd6f)

##### Changed 🔄
- Moved `.motif.query.glytoucan_to_glycan` into `.motif.processing` (eafb218)
- `canonicalize_iupac` will now use `sanitize_iupac` to auto-fix chemical impossibilities in input glycans (407cd6f)

##### Deprecated ⚠️
- XXX

##### Fixed 🐛
- XXX

#### annotate
##### Changed 🔄
- Renamed `clean_up_heatmap` to `deduplicate_motifs` (407cd6f)

#### draw
##### Changed 🔄
- Quantitative highlighting in `GlycoDraw` via the `per_residue` keyword argument will now use individual SNFG-colors instead of a uniform highlight color