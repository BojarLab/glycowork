# Changelog

## [1.11.1]
- `drawsvg` is no longer a dependency: `GlycoDraw` builds its SVG with `glycorender`'s `svgdraw` (`glycorender>=0.5.0`), with identical output and ~1.45x faster drawing (f41edb7)
- initial package import in most cases is now 1.5-2x as fast by lazy-loading `pandas` (6957d6a)
- moderate performance improvements throughout the package (especially glycoproteomics-related) (bc72318)

### motif
#### draw
##### Added ✨
- `GlycanDrawing.as_markdown(size = 16)` returns the drawing as an inline `<img>` (base64 SVG data URI) for Markdown text, e.g., `IPython.display.Markdown(f'... {d.as_markdown()} ...')`, scaled so each monosaccharide symbol is `size` px tall (ba25f21)

##### Changed 🔄

##### Fixed 🐛
- `plot_glycans_excel` leaves the cell of a motif defined by a glyco-regex (such as `Nglycan_complex` in motif-level `get_differential_expression` results) empty instead of failing the whole export, as `annotate_figure` already did ()

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
- Also added mass support for reducine end modifications `reduced_D` (NaBD4 reduction, +HD), `APTS`, `aminoxyTMT`, and `GirardP` (8770c04)
- `sample_prep` keyword arguments now also accept the option of `deuteropermethylated` (8770c04)
- Mass calculation functions etc now support more peptide modifications with their Unimod (PSI-MS) names, such as `'NC[Carbamidomethyl]SK'` (54a3ea2)
- Added `anilide` and `d5-anilide` to the supported sialic acid derivatization (from glycoqueuing), as `GQ` and `GQ-d5` (6a9dae8)
- `'Li+'` as an adduct ion (`get_ion_mzs(adducts = 'Li+')`, `mz_to_composition(extras = ['adduct'], adduct_ions = 'Li+')`) and lithium in `calculate_adduct_mass`, for lithium acetate mobile phases (42fd181)

##### Changed 🔄
- `calculate_adduct_mass` can now also read signed formula parts such as `'C5H6N2-O'` (2aff05a)
- `sialic_derivatization` also weighs a derivatization done before permethylation (`sample_prep = 'permethylated'` or `'deuteropermethylated'`), whose amides are capped on their N-H instead of the carboxyl's methyl ester (`SIALIC_PRODUCT_CAPS`; a lactone raises); a derivatization making the same product of every sialic acid (`GQ`, `('amide', 'amide')`) needs no linkages, and `sialic_linkage_splits(composition, sialic_derivatization)` leaves its compositions unsplit (f0e2839)

#### smiles
##### Added ✨
- Sialic acid methylamides (e.g., SALSA's a2-3 product) are now supported in SMILES generation (5134bdd)

#### draw
##### Changed 🔄
- Better drawing of repeat units with branches (5134bdd)
- `GlycoDraw` `.png` outputs are now much faster (364808d)

#### analysis
##### Changed 🔄
- `get_glycanova`'s Tukey post-hoc p-values are now ~13x faster to compute (16 groups: 1,173 to 97 s) and stay accurate for very small p-values, which were noise below ~1e-11 before (096a01e)

##### Fixed 🐛
- A glycan or motif without any detection in a whole group no longer shifts every other feature: its floor (1e-7) entered each sample's CLR geometric mean and the ALR reference search, so samples with more such features moved as a whole. In `get_differential_expression` and `get_glycanova` such features are now tested on their detection instead (Fisher's exact test, exact McNemar when paired), with Cohen's h (Cramer's V for ANOVA) as effect size and pairwise detection tests as ANOVA post-hoc, and a floored copy of a glycan listed more than once is no longer averaged into its measured copies (c759330)

### glycan_data
#### loader
##### Added ✨
- Added new curated `human_headandneck_O_GPST000316` glycomics datasets to `glycomics_data_loader` (8770c04)

##### Changed 🔄
- Skyline imports in `read_glycomics` can now distinguish blanks, pooled QCs, and calibration standards from real samples (6957d6a)
- `read_glycomics` is now more robust to glycolipidomics input (c7fb126)

##### Fixed 🐛
- Fixed some GlyTouCan IDs being linked to the wrong glycan (6a9dae8)

#### stats
##### Added ✨
- `detection_test` tests whether detection differs between groups (Fisher's exact test, Monte Carlo beyond two groups, or exact McNemar for paired samples) and returns Cohen's h or Cramer's V (c759330)

##### Changed 🔄
- Improved estimate of `mahalanobis_variance` (cb9feeb)
- `impute_and_normalize(return_masks = True)` also returns which cells hold a group's floor and which were detected; `get_additive_logratio_transformation` and `get_procrustes_scores` take `reference`, the rows that may serve as reference (c759330)
- The ALR reference of a two-group comparison (`get_procrustes_scores`) is now the feature whose log-ratio to the geometric mean varies least over all samples (paired: across each pair), resulting in better FDR and +0.14 power ()

##### Fixed 🐛
- `attrs['transform']` of `get_differential_expression` and other `preprocess_data` users no longer reports CLR for an ALR whose reference glycan was listed more than once (10ce1fe)