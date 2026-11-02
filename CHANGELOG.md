# Changelog

## [1.11.2]

### motif
#### draw
##### Added ✨
- `GlycanDrawing.as_markdown(size = 14)` returns the drawing as an inline `<img>` (base64 SVG data URI) for Markdown text, e.g., `IPython.display.Markdown(f'... {d.as_markdown()} ...')`, scaled so each monosaccharide symbol is `size` px tall (ba25f21)
- `GlycanDrawing.as_sticker()` returns any drawing as a die-cut sticker, exactly as `GlycoDraw(..., sticker = True)` would have drawn it, e.g., for slides or pages with a dark background (0820c34)
- frontends with a dark background can set `GLYCOWORK_DARK_DISPLAY=1`, and drawings then display, and inline with `as_markdown`, as die-cut stickers (0820c34)
- frontends that show rich output outside Jupyter (such as an editor's console) can set `GLYCOWORK_RICH_DISPLAY=1`, and `GlycoDraw` and `plot_glycans_grid` then return their drawing instead of opening a matplotlib window (6957d6a)
- `annotate_figure` can now also draws glycan fragments (185471b)

##### Changed 🔄
- a sticker drawing (`sticker = True`) now displays on a transparent background in notebooks, so it looks the same in light and dark themes, instead of on a white box (0820c34)
- Better drawing of repeat units with branches (5134bdd)
- `GlycoDraw` `.png` outputs are now much faster (364808d)

##### Fixed 🐛
- `plot_glycans_excel` leaves the cell of a motif defined by a glyco-regex (such as `Nglycan_complex` in motif-level `get_differential_expression` results) empty instead of failing the whole export, as `annotate_figure` already did (a1ae680)
- `plot_glycans_excel` closes and removes its workbook when a glycan cannot be drawn, instead of leaving it open and half-written (91e264a)
- `per_residue`, `highlight_linkages`, and `highlight_residues` in `GlycoDraw` land on the right arm when another arm carries a wildcard linkage (e.g., `GlcNAc(b1-?)` next to `GlcNAc(b1-2)`) (5134bdd)

##### Deprecated ⚠️

#### processing
##### Added ✨
- `glycan_to_wurcs` converts IUPAC-condensed to WURCS 2.0 in WURCSFramework's canonical order, as GlyTouCan writes it (97% coverage), e.g., for GlyTouCan registration or lookup (322fc99)

##### Changed 🔄
- Improved GlycoCT/GlycoCT XML parsing of phosphodiesters and more exotic monosaccharides (8649553)
- `get_class` can now also deal with aglycones, such as in `Gal(b1-3)GalNAc1Ser` (8649553)
- Improved galactose/sialic acid GlySeeker conversion in `canonicalize_iupac` (48162a4)

##### Fixed 🐛
- `wurcs_to_iupac` now reads residues past the 52nd (`aa`, `ab`, ...), alditols with substituents (`h2122h_3*OSO/3=O/3=O`), anhydro forms (`2,7-Anhydro-Kdo`, `1,5-Anhydro-Glc-ol`), PCho and lactyl groups, and Pse with any acyls on its amines; phosphodiesters are written on the donor (`Rib5P-ol(5-1)`), and `Glc3S-ol` no longer becomes `Glc3-S-ol` (322fc99)
- Fixed some WURCS tokens (322fc99)

#### tokenization
##### Added ✨

##### Changed 🔄
- `glycan_to_mass` and `get_formula` now raise a `ValueError` when modification is set but the glycan has no free reducing end (alditols other than reduction, aldonic acids, inositol, aglycones, sucrose/trehalose-type linkages) ()

##### Fixed 🐛

#### stats
##### Added ✨
- `detection_test` tests whether detection differs between groups (Fisher's exact test, Monte Carlo beyond two groups, or exact McNemar for paired samples) and returns Cohen's h or Cramer's V (c759330)

##### Changed 🔄
- Improved estimate of `mahalanobis_variance` (cb9feeb)
- `impute_and_normalize(return_masks = True)` also returns which cells hold a group's floor and which were detected; `get_additive_logratio_transformation` and `get_procrustes_scores` take `reference`, the rows that may serve as reference (c759330)
- The ALR reference of a two-group comparison (`get_procrustes_scores`) is now the feature whose log-ratio to the geometric mean varies least over all samples (paired: across each pair), resulting in better FDR and +0.14 power (10ce1fe)

##### Fixed 🐛
- `attrs['transform']` of `get_differential_expression` and other `preprocess_data` users no longer reports CLR for an ALR whose reference glycan was listed more than once (10ce1fe)

### network
#### biosynthesis
##### Added ✨
- Added the new `covariates` keyword argument to `get_differential_biosynthesis` to include per-sample covariates (e.g., secretor status, sex, age, batch) (400dd87)
- Added the new `analysis` option of `"steps"` to `get_differential_biosynthesis` to compare biosynthetic step yields between conditions (each motif as the log2 ratio to its tightest containing motif, i.e., how much of the available substrate a step converts) (400dd87)
##### Deprecated ⚠️
