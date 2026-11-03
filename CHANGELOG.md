# Changelog

## [1.12.0]
- All figures are now drawn with `viscose` instead of `matplotlib` and `seaborn`, which are no longer dependencies. The default palette is Aura; `viscose.set_palette` picks another ()
- `**kwargs` of `get_heatmap`, `get_volcano`, and `plot_embeddings` go to `viscose`, and anything it does not know raises a `TypeError` listing what it takes; `plot_embeddings(palette = ...)` names a viscose palette; `get_heatmap(return_plot = True)` returns a `viscose` figure first ()
- Up and down in volcano and MA plots, heatmap and coverage scales, and network nodes and fold-change edges take the palette's colors; the `origin` values of `highlight_network` and `network_alignment` are palette names now ()

### motif
#### tokenization
##### Added ✨
- Added mass support for `glycolate` and `lactate` reducing end modifications, from hypochlorite O-glycan release (bbadeb4)

##### Changed 🔄
- `glycan_to_mass` and `get_formula` now raise a `ValueError` when modification is set but the glycan has no free reducing end (alditols other than reduction, aldonic acids, inositol, aglycones, sucrose/trehalose-type linkages) (6e0e4d9)

##### Fixed 🐛

##### Deprecated ⚠️
