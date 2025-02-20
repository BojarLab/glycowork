# Changelog

## [1.6.0]

### motif
#### processing
##### Added ✨
- GlyTouCanIDs are now another supported nomenclature in the context of Universal Input and can be used as inputs for functions etc, supported via improvements in `canonicalize_iupac` (eafb218)
- Added `sanitize_iupac` to detect and fix chemical impossibilities (like two monosaccharides connected via the same hydroxyl group) and fix it

##### Changed 🔄
- Moved `.motif.query.glytoucan_to_glycan` into `.motif.processing` (eafb218)
- `canonicalize_iupac` will now use `sanitize_iupac` to auto-fix chemical impossibilities in input glycans

##### Deprecated ⚠️
- XXX

##### Fixed 🐛
- XXX

#### annotate
##### Changed 🔄
- Renamed `clean_up_heatmap` to `deduplicate_motifs`