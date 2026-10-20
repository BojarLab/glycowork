# Changelog

## [1.11.1]
- `drawsvg` is no longer a dependency: `GlycoDraw` builds its SVG with `glycorender`'s `svgdraw` (`glycorender>=0.5.0`), with identical output and ~1.45x faster drawing (f41edb7)

### motif
#### draw
##### Added ✨
- `GlycanDrawing.as_markdown(size = 16)` returns the drawing as an inline `<img>` (base64 SVG data URI) for Markdown text, e.g., `IPython.display.Markdown(f'... {d.as_markdown()} ...')`, scaled so each monosaccharide symbol is `size` px tall (ba25f21)

##### Changed 🔄

##### Fixed 🐛

##### Deprecated ⚠️

#### processing
##### Added ✨
- `glycan_to_wurcs` converts IUPAC-condensed to WURCS 2.0 in WURCSFramework's canonical order, as GlyTouCan writes it (97% coverage), e.g., for GlyTouCan registration or lookup (322fc99)

##### Changed 🔄
- Improved GlycoCT/GlycoCT XML parsing of phosphodiesters and more exotic monosaccharides (8649553)
- `get_class` can now also deal with aglycones, such as in `Gal(b1-3)GalNAc1Ser` (8649553)

##### Fixed 🐛
- `wurcs_to_iupac` now reads residues past the 52nd (`aa`, `ab`, ...), alditols with substituents (`h2122h_3*OSO/3=O/3=O`), anhydro forms (`2,7-Anhydro-Kdo`, `1,5-Anhydro-Glc-ol`), PCho and lactyl groups, and Pse with any acyls on its amines; phosphodiesters are written on the donor (`Rib5P-ol(5-1)`), and `Glc3S-ol` no longer becomes `Glc3-S-ol` (322fc99)
- Fixed some WURCS tokens (322fc99)

#### tokenization
##### Added ✨
- Pyridylamination as reducing-end label: `modification = 'PA'` (2-aminopyridine, +78.058 Da) in `composition_to_mass`, `glycan_to_mass`, `get_formula`, `mz_to_composition`, and `mz_to_structures` (2aff05a)
- Also added mass support for reducine end modifications `reduced_D` (NaBD4 reduction, +HD), `APTS`, `aminoxyTMT`, and `GirardP` ()
- `sample_prep` keyword arguments now also accept the option of `deuteropermethylated` ()
- Mass calculation functions etc now support more peptide modifications with their Unimod (PSI-MS) names, such as `'NC[Carbamidomethyl]SK'` (54a3ea2)

##### Changed 🔄
- `calculate_adduct_mass` can now also read signed formula parts such as `'C5H6N2-O'` (2aff05a)

#### smiles
##### Added ✨
- Sialic acid methylamides (e.g., SALSA's a2-3 product) are now supported in SMILES generation (5134bdd)

#### draw
##### Changed 🔄
- Better drawing of repeat units with branches (5134bdd)

### glycan_data
#### loader
##### Added ✨
- Added new curated `human_headandneck_O_GPST000316` glycomics datasets to `glycomics_data_loader` ()

#### stats
##### Changed 🔄
- Improved estimate of `mahalanobis_variance` (cb9feeb)