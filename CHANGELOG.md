# Changelog

## [1.6.0]
- All glycan graphs are now directed graphs (`nx.Graph` --> `nx.DiGraph`), flowing from the root (reducing end) to the tips (non-reducing ends), which has led to code changes in quite few functions. Some functions run faster now, yet outputs are unaffected (03dfad6)

## glycan_data
#### loader
##### Added ✨
- Added `HashableDict` class to allow for caching of functions with dicts as inputs (03dfad6)

##### Changed 🔄
- Refined motif definition of `Internal_LewisX`/`Internal_Lewis_A`/`i_antigen` in `motif_list`, to exclude `LewisY`/`LewisB`/`I_antigen` from matching/overlapping (07c9c12)
- Renamed `Hyluronan` in `motif_list` into `Hyaluronan` (07c9c12)
- Removed `Nglycolyl_GM2` from `motif_list`; it's captured by `GM2` (07c9c12)
- Further curated glycomics datasets stored in `glycomics_data_loader` by introducing the b1-? --> b1-3/4 narrow linkage ambiguities (9eeaa3a)

### motif
#### processing
##### Added ✨
- GlyTouCanIDs are now another supported nomenclature in the context of Universal Input and can be used as inputs for functions etc, supported via improvements in `canonicalize_iupac` (eafb218)
- Added `sanitize_iupac` to detect and fix chemical impossibilities (like two monosaccharides connected via the same hydroxyl group) and fix it (407cd6f)

##### Changed 🔄
- Moved `.motif.query.glytoucan_to_glycan` into `.motif.processing` (eafb218)
- `canonicalize_iupac` will now use `sanitize_iupac` to auto-fix chemical impossibilities in input glycans (407cd6f)
- More GlycoWorkBench sequence variants can now be handled via `glycoworkbench_to_iupac`/`canonicalize_iupac` (9eeaa3a)
- Newly supported WURCS2 tokens: `GalNAc-ol`

##### Deprecated ⚠️
- XXX

#### annotate
##### Changed 🔄
- Renamed `clean_up_heatmap` to `deduplicate_motifs` (407cd6f)

#### draw
##### Changed 🔄
- Quantitative highlighting in `GlycoDraw` via the `per_residue` keyword argument will now use individual SNFG-colors instead of a uniform highlight color (07c9c12)

#### graph
##### Changed 🔄
- Switched `lru_cache` from `glycan_to_graph` to `glycan_to_nxGraph_int` for better performance and fewer opportunities to mess with the cache (03dfad6)

##### Fixed 🐛
- Fixed an edge case in `compare_glycans` in which two identical string glycans returned (True, True) if `return_matches == False` (03dfad6)