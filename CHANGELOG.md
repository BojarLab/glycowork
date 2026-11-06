# Changelog

## [1.12.0]
- All figures are now drawn with `viscose` instead of `matplotlib` and `seaborn`, which are no longer dependencies. The default palette is Aura; `viscose.set_palette` picks another (73bca70)
- `**kwargs` of `get_heatmap`, `get_volcano`, and `plot_embeddings` go to `viscose`, and anything it does not know raises a `TypeError` listing what it takes; `plot_embeddings(palette = ...)` names a viscose palette; `get_heatmap(return_plot = True)` returns a `viscose` figure first (73bca70)
- Glycan graphs now run on a lightweight graph class of their own: `import glycowork` and drawing a glycan no longer load networkx (import plus first `GlycoDraw` 0.37 to 0.19 s, 43 to 33 MB), the graph cache takes 89% less memory (382 to 41 MB for 30,000 glycans), building glycan graphs is 3.2x faster, `compare_glycans` 2.5-2.8x, `subgraph_isomorphism` 1.6-1.9x, `graph_to_string` on strings 1.7x, and `annotate_dataset` 1.6-1.7x. `glycan_to_nxGraph`, `ensure_graph`, and `get_possible_topologies(return_graphs = True)` still return networkx graphs, and every function still takes them (ad36d33)
- Statistics now run on `scipy.special` alone: `glycowork.motif.analysis` imports 35-40% faster and with 24 MB less, imputation fits its censoring model with an analytic gradient (`get_differential_expression` 2.5x faster, and a better optimum on sparse tables), `get_SparCC` with `partial_correlations = True` is 12x faster and counts the controls in its degrees of freedom, `get_SparCC` p-values are exact up to 9 samples and calibrated into the tails beyond, detection tests of absent features are exact at any number of groups, and glycosite p-values use the Cauchy combination, which holds whatever the dependence between a site's glycoforms (54728d1)
- scikit-learn is no longer a core dependency (`glycowork[ml]`, `plot_embeddings`, and `MissForest` still use it) (790a9b8)

### motif
#### graph
##### Added ✨
- Added `glycan_to_glycangraph`, which returns the cached, read-only `GlycanGraph` glycowork's motif code runs on (the nodes, attributes, and neighbor order of `glycan_to_nxGraph`, without networkx), for code that only reads glycan graphs (0f3af98)

#### processing
##### Added ✨
- Added `glycan_to_kcf`, which writes KCF as KEGG GLYCAN does, for KEGG's KCaM search or the RINGS tools (bfa9757)

##### Fixed 🐛
- `kcf_to_iupac` reads KEGG's bare `Fuc`, `Ara`, and `AllNAc` as D, as KEGG means them; both anomers of a sucrose linkage, unknown anomers (`2:1`), phosphoethanolamines, carriers such as `PP-Dol` and `protein`, inositol, and KEGG's spellings (`Cym`, `2,6daraHex`, `L4-en-thrHexA`, `Formyl`) are read as well, so all 1,000 KEGG entries tested now read (bfa9757)
- `kcf_to_iupac` keeps `dHex` as it is, which it turned into `D-Hex` (bfa9757)
- `PDB_TO_IUPAC` now maps `NDG3SO3` to `GlcNAc3S` instead of `GlcNAc3S6S` and adds `BGC6SO3`, `NGA3SO3`, `NGA6SO3`, `NGA4SO36SO3`, and `GAL3SO34SO3` (2a509fe)

#### tokenization
##### Added ✨
- Added mass support for `glycolate` and `lactate` reducing end modifications, from hypochlorite O-glycan release (bbadeb4)

##### Changed 🔄
- `glycan_to_mass` and `get_formula` now raise a `ValueError` when modification is set but the glycan has no free reducing end (alditols other than reduction, aldonic acids, inositol, aglycones, sucrose/trehalose-type linkages) (6e0e4d9)

##### Fixed 🐛

##### Deprecated ⚠️

#### draw
##### Changed 🔄
- Up and down in volcano and MA plots, heatmap and coverage scales, and network nodes and fold-change edges take the palette's colors; the `origin` values of `highlight_network` and `network_alignment` are palette names now (73bca70)

#### annotate
##### Added ✨
- Added `get_motif_atoms` and `get_composition_atoms`, which split a parent motif or glycoform by the children covering it, into parts that sum to the parent (bfa9757)
- Added the `carriers` keyword argument to `get_motif_dag` and `return_counts` to `quantify_motifs` (bfa9757)

##### Fixed 🐛
- `get_motif_dag` proves containment on the measured structures when they are given, since wildcard matching accepted edges such as `Neu5Ac` to `Sia(a2-3)Gal` (bfa9757)

#### analysis
##### Added ✨
- `get_differential_expression`, `get_glycanova`, and `get_pvals_motifs` now also report `Own share`, `Shared share`, and `Share shift` for parent motifs (bfa9757)
- Motif tables now center each sample on a reference that counts a motif at 1 / sqrt(1 + its parents and children), moved a tenth of the way to the median (9afd96d)
- With 6 or more samples in the two largest groups a motif is tested against a change of `min_change` (0.03 log2, TREAT) in `get_differential_expression` and `get_glycanova` (9afd96d)

##### Changed 🔄
- Residuals and redistributions come from an exact partition of each parent, so overlapping children no longer overshoot it; glycoform containment uses the cumulative abundance at or above a composition (bfa9757)
- `correct_multiple_testing` keeps its corrected p-values above 90% significant instead of switching the whole family to Bonferroni, which returned fewer calls for more evidence and did not lower FDR where the log-ratio reference moves; the warning stays (ad36d33)
- `get_equivalence_test` takes `scale_var`, and `get_differential_expression` passes the scale uncertainty to it (ad36d33)
- The scale uncertainty `gamma` now enters `get_differential_expression` and `get_glycanova` as a variance on every sample's log-ratio instead of a random draw per run; results no longer depend on the seed (469cd62)
- Motifs on every glycan in the same number, whose abundance is the total signal by construction, no longer get a test of their own in `get_differential_expression`, `get_glycanova`, `get_time_series`, `get_jtk`, and `get_cosinor`, per glycosite for glycoform motifs. They stay in the log-ratio reference and in the decomposition (469cd62)
- Redistribution in `get_differential_expression` falls back to a permutation test where Hotelling's T2 is undefined (bfa9757)
- `Redistribution p-val` in `get_differential_expression` and `get_glycanova` now also tests each context of a parent, with a share-weighted Simes over moderated F tests on log2(part / rest), joined with the test of all parts by a Cauchy combination, and `get_glycanova` tests all parts by a one-way MANOVA, which is Hotelling's T2 on two groups, instead of PERMANOVA (6bb539b)

##### Fixed 🐛
- Glycoforms without detection in a whole group get a detection test and stay out of their site's log-ratio reference; the floor shifted every other glycoform of the site otherwise (2a509fe)

### glycan_data
#### frame
##### Fixed 🐛
- `GlycoDataFrame.abundance` keeps one row per copy of a duplicated glycan column in the transposed layout, in line with `.glycans` (2a509fe)

### network
#### biosynthesis
##### Fixed 🐛
- `get_maximum_flow` and `get_differential_biosynthesis` no longer change with the interpreter's hash seed (6bb539b)