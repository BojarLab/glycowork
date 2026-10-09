# Changelog

## [1.11.0]
- Removed the `bokeh` dependency from `glycowork`; handled within-package instead (81a910f)
- The installed package is ~50 MB smaller: `v12_sugarbase.json` now ships xz-compressed and `v12_glycan_binding.csv` with four significant digits (dc0e740)
- `df_glycan` loads considerably faster (eac80ee)
- Raised the `pandas` requirement to 2.2.2 (`pandas` 2.1 crashed `get_differential_biosynthesis(longitudinal = True)` and `distance_from_embeddings`, 2.2.0/2.2.1 flooded `GlycoDataFrame` operations with DeprecationWarnings) and the `networkx` requirement to 3.4 (earlier versions crash `generate_graph_features` and `annotate_dataset(feature_set = ['graph'])` with `scipy>=1.16`) (7bc927b)
- Fixed shipped data files (e.g., `common_names.json`) being read in the system encoding, which on Windows broke non-ASCII names such as `GD1α` (ba8b5f7)
- `import glycowork` no longer fails without installed package metadata (source checkout, frozen app) (ba8b5f7)

### motif
#### smiles
##### Added ✨
- Added support for phospho-/sulfodiester in SMILES conversion (e5e8631)
- Added support for acyl chain modifications (01253cb)
- Added support for wildcard alditols such as `HexNAc-ol` (01253cb)
- `glycan_to_smiles` now writes anhydro sugars (e.g., `3,6-Anhydro-L-Gal`, `2,5-Anhydro-Man-ol`, `2,7-Anhydro-Kdo`), heptoses (e.g., `DDGalHep`, `LDIdoHep`, `D-6dAltHep`, `Hep`), aldonic, aldaric, and ulosonic acids (e.g., `Glc-onic`, `Glc2NAc3NAc-aric`, `D-3dThrHex-ulosonic`), and systematic deoxy, keto, and unsaturated names (e.g., `D-4dAraHex`, `D-6dXylHexNAc4Ulo`, `L-4dThrHexA4en`, `8eLeg5Ac7Ac`) ()
- `smiles_to_iupac` now reads open-chain alditols and aldonic acids, anhydro sugars, inositol, pyrophospho- and glycerol phosphodiesters, N-glycosidic linkages, glycosyl halides, and keto groups ()

##### Changed 🔄
- `smiles_to_iupac` now returns each residue in the spelling glycowork uses (e.g., `D-Rha4NFo` instead of `Per4Fo`, `Neu5Ac9NAc` instead of `Kdn5NAc9NAc`, `D-Apif` instead of `D-Api`) ()

##### Fixed 🐛
- Fixed `(2R,3R) L-threo` to `D-erythro (2S,3R)` sphingosine as a default for ceramide (01253cb)
- Fixed amino acid configuration sometimes not defaulting to `L` (01253cb)
- Fixed handling of `1N`-modified monosaccharides such as `GlcNAc1N` and halogen-substituted monosaccharides such as `Gal6F` (01253cb)
- Fixed alditols engaging in glycosidic linkages (01253cb)
- Various other fixes for more exotic modifications (01253cb)
- Fixed inverted stereochemistry of `All`/`Allf` and of `Lac`, `NAla`, `NSer`, and `NThr` substituents (01253cb)
- Fixed `smiles_to_iupac` misreading rings fused to a residue, such as pyruvate ketals (01253cb)
- Fixed `glycan_to_smiles` placing a glycosidic bond with unknown position onto the amine of an amino sugar instead of a hydroxyl (a34092c)
- Fixed `smiles_to_iupac` dropping (or, with `strict = True`, refusing) pyruvate ketals bridging two positions, such as `Gal4Pyr6Pyr` (ba8b5f7)
- Fixed `IUPAC_to_SMILES` crashing on an empty cell (now `''`); `glycan_to_smiles`/`glycan_to_molecule` raise an informative error for a list (ba8b5f7)
- Fixed `DDAltHep` and `LDAltHep` being built on L-altrose, the swapped anomers of `Abef`, and the alpha anomers of `Pse`, `Aci`, and `Fus`, which IUPAC's reference atom puts on the other side for an L ring; `Aci` is now L ()
- Fixed `glycan_to_smiles` putting an O-prefixed group of unknown position (`GlcNOMyr`, `GalOAcN`) onto the amine ()

#### draw
##### Added ✨
- Added the new `plot_glycans_grid` function to draw glycans with a shared symbol size into a grid (0967d8a)
- `GlycoDraw` now also accepts alternative composition formats (e.g., `{'Hex': 5}`) (0967d8a)
- `GlycoDraw` now accepts lists as input, which are fed into the new `plot_glycans_grid` (0967d8a)
- `plot_glycans_excel` can now pass on styling keyword arguments (e.g., `compact`, `vertical`, etc) to `GlycoDraw` (0967d8a)
- `GlycoDraw` drawing objects have a new `.save()` method, to save them as `.svg`, `.pdf`, or `.png` (0967d8a)
- Added SNFG symbols for the furanoses `Hexf`, `Penf`, `Rhaf`, `6dTalf`, `Parf`, `Sorf`, and `Psif` (0967d8a)
- `glycan_col_num` in `plot_glycans_excel` can now also be a column name (0967d8a)
- Added the `highlight_residues` keyword argument to `GlycoDraw` to have `highlight_motif` behavior for manually specified residues (7bc927b)

##### Changed 🔄
- Black elements in `.svg` `GlycoDraw` outputs are now also charcoal (`#1C1917`), just like any other format (0967d8a)
- Symbols in vertical mode are no longer rotated in `GlycoDraw`, in accordance with SNFG (0967d8a)
- `GlycoDraw` outputs now also carry metadata in notebook outputs as well as Excel cells (0967d8a)
- The `folder_filepath` argument in `plot_glycans_excel` can now also be used to specify a desired output file name (0967d8a)
- Better layouting of complex floating bits in `GlycoDraw` (0967d8a)
- Modification and linkage labels in `GlycoDraw` that would collide with symbols, bonds, or other labels are now shrunk or moved (0967d8a)
- `GlycoDraw` now raises an informative error for glyco-regex motif names (e.g., `Nglycan_hybrid`), which match a family of structures, pointing to `highlight_motif` (0967d8a)
- `plot_glycans_excel` now also embeds every drawing as a vector graphic, displayed by Excel 2016+ (0364cf7)
- `per_residue` and `highlight_linkages` in `GlycoDraw` now also accept arrays and Series, and `mono_list` in `draw_chem2d`/`draw_chem3d` a single string (79c0ac9)

##### Fixed 🐛
- Fixed `Altf` and `Tagf` not being recognized as furanoses by `GlycoDraw` (0967d8a)
- Fixed `NS` not being recognized as a joint modification by `GlycoDraw` (0967d8a)
- Fixed `draw_method='chem2d', filepath='x.png'` not saving a `png` (0967d8a)
- Fixed blurriness of glycan drawings in `plot_glycans_excel` (0364cf7)
- Fixed `show_linkage = False` still showing linkages in floating bits (0967d8a)
- Fixed `reverse_highlight` hiding floating bits, brackets, and repeat labels (0967d8a)
- Fixed `per_residue` values landing on the wrong residues in glycans with anchored floating bits (79c0ac9)
- Fixed `per_residue` and `highlight_linkages` in `GlycoDraw` landing on the wrong residues/linkages of glycans with floating parts whose sequence gets reordered for drawing (ba8b5f7)
- Fixed `per_residue` and `highlight_linkages` in `GlycoDraw` being ignored on floating and anchored bits; identical floating bits are only merged into one "2x" symbol if their values agree (ba8b5f7)
- Fixed `plot_glycans_excel` crashing on a Series or list of glycans (ba8b5f7)

#### tokenization
##### Added ✨
- Added `get_ion_mzs` to support more complex adducts in `mz_to_composition` (a512f5d)
- Added the new `adduct_ions` keyword argument to `mz_to_composition` to specify a list of permitted adduct ions (a512f5d)
- `calculate_adduct_mass` now also supports F/Br/I halogen atom masses (a34092c)
- `match_composition_relaxed`, `compositions_to_structures`, and `mz_to_structures` now also take `glycan_class = 'all'` (a34092c)
- `glycan_to_mass` now also takes compositions (dicts, or strings such as `'H5N4F1'`) ()

##### Changed 🔄
- `get_core` and `get_modification` can now better deal with cases such as `Neu5,9Ac2` (0967d8a)
- `map_to_basic` is now cached, making functions that call it repeatedly (e.g., `structure_to_basic`) about 2x faster (eac80ee)
- `mz_to_composition` and related functions are now faster (eac80ee, 2703d28)
- `condense_composition_matching` now condenses better (cac1c78)
- Acetonitrile and Trifluoroacetic Acid are now correctly handled as neutral adducts in composition matching (afb2752)
- `mz_to_composition` with `extras = ['adduct']` now also considers ions carrying one adduct per charge, such as `[M+2Na]2+` (a512f5d)
- `calculate_adduct_mass` and `composition_to_mass` now raise an error for malformed formulas, unknown elements, or unknown `sample_prep`/`mass_value` instead of silently ignoring them (a34092c, cac1c78)
- More precise element masses in `calculate_adduct_mass` (cac1c78)
- `glycan_to_composition` now returns `{}` instead of a bare-core composition when a residue carries a substituent it cannot count (e.g., `Pyr`, amino `N`, `Aep`, `Gro`, `Lac`, acyl chains, `-onic`, anhydro), so `glycan_to_mass` raises instead of returning a wrong mass (ba8b5f7)
- `glycan_to_composition` now says when it was given a composition (dict or string like `'H5N4'`) instead of a sequence (ba8b5f7)
- `get_ion_mzs`, `mz_to_composition`, and `mz_to_structures` now accept a single adduct, `filter_out`, or `deprioritized` string and a single m/z (ba8b5f7)

##### Fixed 🐛
- Fixed handling of `adduct` and `mass_tag` with multiply-charged glycans in `mz_to_composition` (a34092c)
- Fixed handling of amino acid and ceramide aglycones (e.g., `GalNAc1Ser`, `GlcNAc1Asn`) in `glycan_to_composition` (a34092c)
- Fixed azido sugar masses (cac1c78)
- Fixed handling of reduced peracetylated glycans in `composition_to_mass` (cac1c78)
- Fixed average masses of reducing-end modifications in `composition_to_mass`, which used monoisotopic values (cac1c78)
- Fixed `glycan_to_composition` missing O-acetyls at positions 1, 5, and 8 (e.g., `Neu5Ac8Ac`, `Kdn5Ac`) and counting repeated lactones only once (cac1c78)
- Fixed the `Formate`, `HCO3-`, and `Cl-` adduct masses used by `mz_to_composition` (a512f5d, afb2752)
- Fixed `mask_rare_glycoletters` masking parts of glycoletters (e.g., `LDManHep` to `LDManMonosaccharide`) and treating uncertain linkages such as `b1-3/4` as monosaccharides (a34092c, cac1c78)
- Fixed `stemify_dataset` ignoring its rarity filter for single-residue glycans (cac1c78)
- Fixed `glycan_to_composition` raising on linkages that `canonicalize_iupac` writes but `lib` lacks (e.g., `Kdo(?2-4)`), on floating substituents (e.g., `{OS}`, `{6P}`), and on modified residues with an aglycone (e.g., `GalNAc1Thr6S`) (ba8b5f7)
- Fixed `glycan_to_mass` giving glycans with alditols (`-ol`, at the reducing end or inside the chain such as the `Rib5P-ol` of teichoic acids) the mass of the free sugar (ba8b5f7)
- Fixed `composition_to_mass` overestimating permethylated and peracetylated masses of O-acetylated and methylated glycans, and permethylated/peracetylated `glycan_to_mass` of inositol-containing glycans (ba8b5f7)

#### processing
##### Added ✨
- The LINUCS nomenclature is now also supported in Universal Input/`canonicalize_iupac` via the new `linucs_to_iupac` parser (e1c807a)
- StrucGP structure codes (e.g., `A2B2C1D1E2F1fedD1E2edcbB5ba`) are now also supported in Universal Input/`canonicalize_iupac` via the new `strucgp_to_iupac` parser, which infers monosaccharides and linkages from N-glycan position and StrucGP's arm order (99ad40e)
- Universal Input/`canonicalize_composition` now also supports composition nomenclatures from Byonic, FragPipe, GlycoMod, GlyHunter, GlyCombo, LaCyTools, MassyTools, GlycoGenius, and GlycReSoft (e1c807a, f936c23)
- Expanded GlyTouCan ID coverage (dc0e740)
- Support heptoses in WURCS conversion (dc0e740)
- Added `strict` keyword argument to `canonicalize_composition` to raise an error if any non-valid component is present (4c31c0a)
- Added `glycan_class` keyword argument to `parse_glycoform` and `process_for_glycoshift`, to prevent annotating O-glycoproteomics data with N-glycan features by `infer_features_from_composition` (4c31c0a)
- Added `peptide` keyword argument to `composition_to_mass` and `glycan_to_mass`, to calculate glycopeptide masses (4c31c0a)
- Added `split_glycoform_id` to split glycoproteomics row labels (`protein_site_glycan`, including joint sites like `31+40`, ambiguous sites like `226/229`, and per-glycosite motifs) into glycosite and glycan (4c31c0a)
- Added `glycan_to_oxford` to name N-glycans in Oxford nomenclature (e.g., `FA2G2S(6)2`), refusing structures whose name would read back as a different composition (ba8b5f7)
- Added `glycan_to_glycoct` to write GlycoCT condensed records, e.g., for GlycoWorkbench, glypy, or GlyTouCan registration (ba8b5f7)
- Added `glycan_to_glycam` to write GLYCAM condensed sequences, e.g., for the GLYCAM-Web carbohydrate builder, with the `reducing_anomer` keyword argument for the reducing-end anomer (ba8b5f7)
- Added `glycan_to_iupac_extended` to write IUPAC-extended sequences as GlyTouCan does (e.g., `α-D-Neup5Ac-(2→3)-β-D-Galp-(1→4)-?-D-GlcpNAc-(1→`) (ba8b5f7)
- Added the `string_format` keyword argument to `canonicalize_composition` to write compositions as Byonic/MSFragger-Glyco (`HexNAc(4)Hex(5)Fuc(1)NeuAc(2)`) or GlycReSoft (`{Fuc:1; Hex:5; HexNAc:4; Neu5Ac:2}`) glycan database entries (ba8b5f7)
- Added the IgG names `G0N`, `G2FS2`, `G2NS2`, `G2FNS2`, and the S1 spellings (e.g., `G2S1`, `G2FS1`, `G1NS1`) to `canonicalize_iupac`, which previously misread or rejected them (ba8b5f7)

##### Changed 🔄
- `canonicalize_composition` now also accepts dictionary compositions as inputs (0967d8a)
- `canonicalize_iupac` is more robust to nomenclature variations (e1c807a)
- Moved `PDB_TO_IUPAC` from `glycontact` up into `glycowork` to facilitate atom-coloring in `.draw.draw_chem3d` without `glycontact` dependencies (0a28132)
- Refined `infer_features_from_composition` output (342a605)
- `get_motif_dag` and `quantify_motifs` now also support glycoproteomics data, if glycan sequences are available (4c31c0a)
- `canonicalize_iupac` now gives `Neu5Gc(a2-?)` the same defaults as `Neu5Ac(a2-?)` (`a2-3/6`, or `a2-8` onto Neu) (0364cf7)
- `wurcs_to_iupac` now raises an informative error for repeating units, linkages bridging more than two residues, and cyclic structures (dc0e740)
- `max_specify_glycan` now raises an error for species not in `df_species` (342a605)
- `canonicalize_composition` now reads an IUPAC-condensed sequence as its composition instead of as residue tokens; `strict = True` still rejects sequences (ba8b5f7)
- `get_possible_linkages`, and with it `compare_glycans`/`subgraph_isomorphism`, now raises an informative error for a malformed linkage such as `?3/6` (ba8b5f7)

##### Fixed 🐛
- `sanitize_iupac` no longer flags phosphodiesters (e1c807a)
- Fixed handling of variantly capitalized monosaccharides in KCF (e1c807a)
- Fixed handling of `NS` in GlycoWorkbench parsing (43d7a40)
- Fixed `glycoctxml_to_iupac` dropping substituent positions such as in GlcNAc6S (342a605)
- Fixed `wurcs_to_iupac` leaving `|` in its output for attachment alternatives spanning several residues (e.g., `d4|d6|g4|g6`), which now give a floating part (dc0e740)
- Fixed lone residues with positioned substituents (e.g., `Man2F6P`) being read as Oxford nomenclature by `canonicalize_iupac` (dc0e740)
- Fixed `canonicalize_iupac` stripping modified residues after a dash (e.g., `2,5-Anhydro-Tal6P`) as linkers, turning library residues (e.g., `D-Rha4NLac`) into unknown ones while sorting modifications, and giving a different result on a second pass when sibling branches claimed the same position (a34092c)
- Fixed `oxford_to_iupac` reading non-sialylated hybrids (e.g., `M5A1G1`) as complex glycans, and sulfate counts (e.g., `A2G2Sulf2`) as galactoses (342a605, a34092c)
- Fixed `glytoucan_to_glycan(revert = True)` returning NaN for glycans without a GlyTouCan ID instead of reporting them as not found (a34092c)
- Fixed functions accepting compositions (e.g., `composition_to_mass`) also trying to read other string arguments such as `mass_value` as compositions (342a605)
- Fixed `max_specify_glycan` specifying `GlcNAc(b1-?)GlcNAc` outside of the N-glycan core (342a605)
- Fixed `process_for_glycoshift` multiplying rows of datasets with a duplicated index (eac80ee)
- Fixed `canonicalize_iupac` reading the a/b ending a residue name (`Rib1N5PP-ol`, `GlcNAcA3NFoAla6N`) as an anomer, turning the comma of multiple-deoxy prefixes (`DL-3,9dGalNon5NAc7NAc-ulosonic`) into a dash, and turning `Bac1N2Ac4Ac`/`Glc1EtN2Ac` into `Bac1NAc4Ac`/`GlcNAc1Et` (ba8b5f7)
- Fixed `canonicalize_iupac` not being idempotent when a sibling position clash made a sialic acid linkage uncertain, and on 12 common names such as `GM2` and `SM4` (ba8b5f7)
- Fixed the common name `G1FS` lacking its core fucose, and `canonicalize_iupac` reading `G0F-N`/`G0-N`/`G1F-N` as the bisected `G0FN`/`G1FN` (ba8b5f7)
- Fixed `canonicalize_iupac` crashing on GlycoWorkbench structures with a labelled reducing end such as `2AB` (ba8b5f7)
- Fixed `oxford_to_iupac` ignoring the stated core fucose linkage in `F(3)` and `F(6)`, and dropping the core xylose in names with a bisecting GlcNAc (e.g., `XA2B`) (ba8b5f7)
- Fixed `glycam_to_iupac` failing on GLYCAM-Web's `Ac` derivative code (e.g., `DGalp[4Ac,2S,3Me]b1-OH`) (ba8b5f7)
- Fixed `glycam_to_iupac` garbling pyranose ketoses (`DFrupb2-...`), two-digit acceptors (`Neu5Gc(a2-11)`), the order of sialic acid derivatives, and `b2-OME` (ba8b5f7)
- Fixed `glycoct_to_iupac` dropping alternative linkage positions (`3|6` now `3/6`), raising on three alternatives (`2|3|6`), not knowing stem codes such as `lglc`, `drib`, `dtal`, and `lgul`, losing the `-ol`/uronic/6-deoxy names of residues with several modifiers (`Rha-ol`, `GalA-ol`), writing every N-glycolyl as `5Gc`, ordering sialic acid substituents by listing order, reading O-substituted `Kdn` as `Neu`, and not reading O-glycolyl (ba8b5f7)
- Fixed IUPAC-extended input keeping the pyranose `p` inside residues with substituents (`MurpNAc`) or taking it from substituents ending in `p` (`Gal6Aep`), and not reading residues without D/L and anomer (`?-Kdop-(2→`) or glycans starting with an alditol (ba8b5f7)
- Fixed `canonicalize_iupac` keeping the default `D-`/`L-` prefix of rarer trivial names (e.g., `Abe`, `Dig`, `Col`, `Pse`, `Dha`, `Ko`, `Leg`) (ba8b5f7)
- Fixed `linearcode_to_iupac` giving `Kdo` and `Fruf` a C1 donor and garbling unknown anomers or positions, and `canonicalize_iupac` returning a lone LinearCode residue such as `GN` unchanged (ba8b5f7)
- Fixed `canonicalize_iupac` giving sialic acids of unknown position `a2-3/6` instead of `a2-8` on a Neu acceptor behind or inside a branch (e.g., `Neu5Ac(a2-?)[Gal(b1-3)]Neu5Ac`) (ba8b5f7)

##### Deprecated ⚠️
- Removed the `degrees` keyword argument from `glycoct_build_iupac`; handled automatically (342a605)

#### annotate
##### Added ✨
- Added `quantify_dag_steps` to re-express motif abundances as non-redundant, scale-free biosynthetic step yields (log2 ratio of each motif to its tightest containment parent in `get_motif_dag`) (51c1fc6)

##### Changed 🔄
- Motif annotation is now generally faster (eac80ee)
- Motif annotations with `feature_set` `terminal` are now ~10x faster (eac80ee)
- `annotate_dataset` now counts `custom_motifs` (sequences, `r`-prefixed glyco-regexes, or motif names) in the same pass as the known motifs, keeps each column labeled as given, and raises an error for entries that are none of these (79c0ac9)
- `annotate_dataset` now keeps a glycan and its alditol (`-ol`) as separate rows (79c0ac9)
- `annotate_dataset` now raises an informative error for an empty list or a composition dict (ba8b5f7)

##### Fixed 🐛
- Fixed `annotate_dataset`, `annotate_glycan_topology_uncertainty`, and `get_k_saccharides` counting motifs whose stated PTM positions differ from the glycan's (e.g., `Gal6S(b1-4)GlcNAcOS` in `Gal3S(b1-4)GlcNAc6S`, or `Terminal_Glc3Ac` in glycans with `Glc6Ac` and an `Ole` residue) as soon as either side contained an `O` (9576826)
- Fixed labeling of `size_branch` `feature_set` bins (79c0ac9)
- Fixed `get_k_saccharides(up_to = True)` counting monosaccharides inside other names (e.g., the `Fuc` in `D-Fuc` or the `Hep` in `LDManHep`) (79c0ac9)
- Fixed `group_glycans_sia_fuc` not treating `Kdn` and `Sia` as sialic acids (79c0ac9)
- Fixed `get_molecular_properties` dropping halogens from the formula of residues such as `Gal6F` (a34092c)
- Fixed `annotate_glycan` missing motifs at a reducing-end alditol, now matching `annotate_dataset` (ba8b5f7)

#### analysis
##### Added ✨
- Added `get_cosinor` to analyze circadian glycomics data via Cosinor analysis (6252915)
- `get_biodiversity` on glycoproteomics data (using its new `glycoproteomics` keyword argument) now reports per-glycosite microheterogeneity (4c31c0a)
- Added `glycan_class` keyword argument to `get_glycoshift_per_site`, to prevent annotating O-glycoproteomics data with N-glycan features by `processing.infer_features_from_composition` (4c31c0a)

##### Changed 🔄
- `get_glycanova` is now faster (eac80ee)
- Every analysis function that takes a file path now also reads exports of glycomics and glycoproteomics tools (via `read_abundances`), and `get_differential_expression`/`get_glycanova` switch to glycoproteomics mode by themselves for glycoproteomics exports (43d7a40)
- Every analysis function that takes a file path now also takes the name of a dataset shipped with glycowork (e.g., `get_differential_expression('human_serum_bacteremia_N_PMID33535571')`), which loads with its contrasts (6252915)
- `get_SparCC` now uses `spearman_exact_pvals` at small n (<=8) to avoid anticonservative bias (51c1fc6)
- `get_jtk` now imputes with `impute_biosynthetic` (with a draw of its prediction error for every imputed value) instead of `MissForest` (8124f7b)
- The `glycoproteomics` keyword argument of `get_differential_expression`, `get_glycanova`, `get_time_series`, `get_jtk`, `get_cosinor`, and `get_biodiversity` now defaults to `None`, which switches it on whenever every row label is `protein_site_glycan` (4c31c0a)
- With `motifs = True`, glycoproteomics data are now analyzed as motifs per glycosite (`protein_site_motif`) (4c31c0a)
- `preprocess_data(glycoproteomics = True)` now raises an informative error when no glycosite has enough measured glycoforms to test (99ad40e)
- Analysis functions and `quantify_motifs` now also accept glycans held in the index instead of reading the first sample as the glycans (ba8b5f7)
- `get_volcano` now works on glycoproteomics (per-glycosite) results, using the effect size, and `get_ma` plots their glycoform table (ba8b5f7)
- More informative errors for common group mistakes: a one-sample or empty group in `get_differential_expression`, unequal groups with `paired = True`, a sample in both groups, more group labels than samples, and unknown sample names or wrong-length labels in `get_lectin_array` (ba8b5f7)
- `get_differential_expression(sets = True)` now says when no glycan set correlates above `set_thresh` instead of returning an empty table silently (ba8b5f7)
- `get_glycanova` only runs Welch's ANOVA when `moderate_variance = False` (output unchanged, no more warnings for one-sample groups) (ba8b5f7)

##### Fixed 🐛
- `get_glycanova` no longer runs a redistribution PERMANOVA without within-group degrees of freedom (one sample per group), which divided by zero (8124f7b)
- Fixed `get_pca`, `get_pcoa`, and `get_glycanova` pairing contrast labels with the wrong samples when a `GlycoDataFrame` has sample columns without a contrast label (a34092c)
- Fixed `get_biodiversity` richness counting glycans that were only floored (zero in a whole group) as present (4c31c0a)
- Fixed the class order of multi-group `get_roc` depending on `PYTHONHASHSEED` (a34092c)
- Fixed `get_heatmap` and `get_distance_matrix` transposing a frame with a named glycan column when sample names contain brackets (e.g., `mouse_taysachs_O`), which broke `get_pcoa` and motif heatmaps (ba8b5f7)
- Fixed `get_biodiversity` (and `impute = False` runs on ALR data) crashing in the Procrustes search, and `get_time_series(impute = False)` returning NaN trends, on zeros (ba8b5f7)
- Fixed `get_time_series` transposing composition-labelled tables (e.g., `Hex5HexNAc4` rows) (ba8b5f7)
- Fixed `get_lectin_array` crashing on the shipped `bacteria_PMID20973590` array and reading 0-based indices as 1-based when a group starts with 1 (e.g., `[1, 3, 5]` vs `[0, 2, 4]`), and explained why `transform = 'log2'` fails on data already on a log scale (ba8b5f7)
- Fixed `get_roc(multi_score = True)` crashing when the L1 step zeroes every coefficient (ba8b5f7)
- Fixed `get_jtk` and `get_biodiversity(circadian = True)` crashing on a single period such as `periods = 24` (ba8b5f7)
- Fixed `get_volcano` dropping corrected p-values that underflow to 0 (ba8b5f7)

##### Deprecated ⚠️
- Removed the `circadian_timepoints`, `circadian_periods`, `circadian_interval`, and `circadian_replicates` keyword arguments of `preprocess_data`; imputation no longer uses sample times (8124f7b)

#### graph
##### Added ✨
- Added the new `subsumes` keyword argument to `compare_glycans`, to support asymmetric wildcard matching (glycan_b is glycan_a or a more specific version of it, but not vice versa) (9576826)

##### Changed 🔄
- `compare_glycans`, `subgraph_isomorphism`, and `ensure_graph` (and with it `annotate_glycan`, `get_terminal_structures`, `largest_subgraph`) raise an informative error for a composition dict, a list, or `None` (ba8b5f7)

##### Fixed 🐛
- Fixed `compare_glycans("Gal3S(b1-4)GlcNAcOS", "Gal6S(b1-4)GlcNAc6S")` returning `True` (PTM wildcards made specific PTMs non-comparable; same in `subgraph_isomorphism`) (9576826)
- Fixed `compare_glycans` returning `False` for identical glycans whose floating parts were written in a different order (342a605)
- Fixed `largest_subgraph` returning disconnected parts and dangling linkages; it now finds the largest common subtree (342a605)
- Fixed `subgraph_isomorphism` with `termini_list` ignoring positions when given a glycan graph built without termini (a34092c)
- Fixed `graph_to_string` ordering some branches by how the input was written, so its output was not always canonical (a34092c)
- Fixed `compare_glycans` and `subgraph_isomorphism` matching residues whose stated PTM positions disagree when only one side states all of them (e.g., `Glc3Ac4PGro` and `Glc6AcOPGro`) (ba8b5f7)

#### regex
##### Fixed 🐛
- Fixed `get_match` matching residues with differently stated PTM positions as soon as the glycan or the pattern contained a PTM wildcard (9576826)
- Fixed `motif_to_regex` and glyco-regexes for residues with dashes in their name (`Glc-ol`, `3,6-Anhydro-Gal`, `Thre-onic`) and for two-digit acceptor positions (`Fuc(a1-11)Neu5Gc`) (ba8b5f7)

#### query
##### Changed 🔄
- `get_insight` now canonicalizes its input, so any notation supported by `canonicalize_iupac` is found (ba8b5f7)

### ml
#### inference
##### Changed 🔄
- `glycans_to_emb` can now also be used with `GIFFLAR`-type models (27dccf6)

##### Fixed 🐛
- Fixed `glycans_to_emb(rep = False)` always predicting the first class with single-output (binary) models (a34092c)

#### processing
##### Fixed 🐛
- Fixed `dataset_to_graphs` failing on glycans with anchored floating parts (eac80ee)

#### models
##### Added ✨
- A trained `GIFFLAR` checkpoint is now available (for embeddings etc) via `prep_model` (f568dc2)

##### Changed 🔄
- `GIFFLAR`-type models are now more accurate and almost twice as fast to train (Kingdom prediction MCC 0.835 to 0.876, macro-F1 0.758 to 0.868; Phylum MCC 0.746 to 0.812, macro-F1 0.579 to 0.746), via two-layer GIN updates with learnable self-weights, residual updates, per-node-type pooling, linkage-aware monosaccharide edges, and 4 instead of 8 layers (27dccf6)
- `GIFFLAR` now takes a `dropout` keyword argument (27dccf6)
- Improved the `SweetNet` model architecture via BatchNorm plus residual updates after each convolution, with mean and max pooling (f568dc2)
- The trained `SweetNet` checkpoint via `prep_model` has been updated (f568dc2)

#### model_training
##### Changed 🔄
- Lowered default `rho` for `SAM` from 0.5 to 0.05 (f568dc2)
- `train_model` now also accepts `training_setup`'s mode names (`'multiclass'`, `'binary'`) (a34092c)

##### Fixed 🐛
- Fixed `train_model` averaging per-batch metrics (MCC, AUROC, LRAP, R2), which biased them; every metric is now computed once per epoch (f568dc2)
- Fixed gradient clipping under `SAM` acting on the perturbation step, where it had no effect, instead of the update step (f568dc2)
- Fixed `SAM` failing when some parameters receive no gradient in a step (26eb502)
- Fixed `WarmupScheduler` handing the validation loss to schedulers other than `ReduceLROnPlateau` as their epoch, and `train_model` not casting multilabel targets to float (a34092c)
- Fixed `get_mismatch` indexing `y_test` by label instead of position (a34092c)

### glycan_data
#### stats
##### Added ✨
- Added `impute_biosynthetic`, a bespoke glycomics imputer that models log-abundances with per-sample activities of each biosynthetic step (monosaccharide and monosaccharide-linkage counts) plus a low-rank term, stacks it with stable-ratio partner glycans, and corrects for detection-limit censoring with a fitted selection model (f568dc2)
- Moved `pvca` and `cosinor_fit` up from `glycoforge` into `glycowork` (6252915)
- Added `spearman_exact_pvals` to avoid anticonservative bias of Spearman's t approximation at small n (51c1fc6)

##### Changed 🔄
- `impute_and_normalize`, and with it every function with an `impute` keyword, now imputes with `impute_biosynthetic` instead of `MissForest`: 33-46% lower log2 RMSE than `MissForest` across 41 glycomics datasets, 12-23% lower than MICE, and 94% instead of 68% of true differential hits recovered (f568dc2, 8124f7b)
- Cells floored because a whole group is zero no longer enter imputation as measurements (f568dc2)
- Made sure `replace_outliers_winsorization` does no longer removes zero values (that's the job of imputation) (8124f7b)

##### Fixed 🐛
- Fixed `bh_adjust` and `TST_grouped_benjamini_hochberg` turning every adjusted p-value of a family into NaN when one p-value was NaN; NaN p-values are now left out (a34092c)
- Fixed `hsic` returning NaN for a constant variable (a34092c)
- Fixed `get_glycoform_diff(level = 'protein')` cutting accessions that contain underscores (e.g., `A1AG1_HUMAN`) (4c31c0a)
- `get_additive_logratio_transformation` now floors zeros at 1e-7, like the CLR route, instead of taking log2(0) (ba8b5f7)

##### Deprecated ⚠️
- Removed the `timepoints`, `periods`, `interval`, and `replicates` keyword arguments of `impute_and_normalize`, which only fed `MissForest` (8124f7b)

#### data_entry
##### Added ✨
- Added `read_glycoproteomics`, which reads the native output of FragPipe/MSFragger-Glyco and O-Pair, pGlyco3 and pGlycoQuant, Byonic and Byologic, GlycReSoft, MetaMorpheus O-Pair, Glyco-Decipher, StrucGP (at structure level via its structure codes), and PEAKS GlycanFinder into a `protein_site_composition` x sample `GlycoDataFrame`, ready for `get_differential_expression(glycoproteomics = True)` and `get_glycoshift_per_site` (f936c23)
- Added `read_glycomics`, which reads Skyline reports, LaCyTools and MassyTools summaries, and GlycoGenius, GlyHunter, GlycReSoft, GlycoWorkbench workspaces (.gwp) and annotated peak lists (.gwa), Thermo Compound Discoverer compound tables, and CandyCrunch tables into a glycan x sample `GlycoDataFrame`, with every label turned into a canonical sequence or composition and GlycoGenius groups kept as contrasts (f936c23, 43d7a40)
- Added `read_abundances`, which every analysis function uses to read a file path (tool exports via `read_glycomics`/`read_glycoproteomics`, else a plain table) or the name of a dataset shipped with glycowork (43d7a40, 6252915)
- `read_glycoproteomics` also reads pGlycoQuant TMT quantification, with one sample per reporter channel (69d0a3e)

##### Changed 🔄
- `read_glycoproteomics`, `read_glycomics`, and `read_abundances` now read semicolon-separated CSVs with decimal commas and cp1252-encoded files, as written by European Excel or Skyline (ba8b5f7)
- `read_glycoproteomics` and `read_glycomics` now accept any iterable of files, e.g., `Path.glob('*/psm.tsv')`, and `read_glycomics` takes the first sheet of a workbook with a recognized tool export (ba8b5f7)
- `read_glycomics` now says that glycans belong in the first column when given a frame with glycans in its index (ba8b5f7)

##### Fixed 🐛
- Fixed own tables with `Site`/`Glycan` or `Protein Name`/`Position`/`Composition` columns being read as Glyco-Decipher site tables or Byonic lists by `read_glycoproteomics`/`read_abundances` (ba8b5f7)
- Fixed decoy proteins (`REV_`, `rev_`, `DECOY_`, `#DECOY#`, `XXX_`) of all engines becoming glycoforms in `read_glycoproteomics`, and contaminant labels (`contam_`, `CON__`) giving wrong accessions (ba8b5f7)
- Fixed empty protein cells becoming the accession `nan`, and pGlyco, Byologic, and MetaMorpheus tables without their run column coming back empty or with a sample named `None`, in `read_glycoproteomics` (ba8b5f7)

#### loader
##### Added ✨
- Datasets above 1 MB now ship as xz-compressed CSV (`.csv.xz`), which all data loaders read without special treatment (99ad40e)
- Added new curated glycoproteomics datasets: `tomato_fruit_mnsI1_N_PMID41017156`, `human_hek293_stt3_N_PMID36139350`, `human_hek293_surface_N_PMID39930009`, `mouse_tissues_N_PMID39930009`, `mouse_brain_neurodegeneration_N_PMID40593524`, `mouse_macrophages_infection_N_PMC8416091`, `human_serum_igg_liverdisease_N_PMID36879659`, `human_fibroblasts_srd5a3cdg_N_PMID39360848`, `human_fibroblasts_ngly1cddg_N_PMID36102038`, `schistosoma_mansoni_sex_N_PMID41545360`, and `schistosoma_mansoni_sex_O_PMID41545360` to `glycoproteomics_data_loader` (99ad40e)

##### Changed 🔄
- Streamlined row labels of `human_milk_N_PMID34087070` glycoproteomics dataset (69d0a3e)
- Re-curated `human_keratinocytes_N_PMID37956981` (absolute intensities instead of per-peptide shares) and `sorghum_N_PMID39137587` (per-run intensities of all glycopeptides instead of log2 fold changes of the deregulated ones) glycoproteomics datasets from their sources (69d0a3e)

##### Fixed 🐛
- Fixed `groupby(...).apply` failing on `GlycoDataFrame`s such as `df_species` (eac80ee)
- Removed a blank trailing row from `lectin_array_bacteria_PMID20973590` (ba8b5f7)

##### Deprecated ⚠️
- Removed `human_lipoproteins_PMC9218963` glycoproteomics dataset (no per-sample values) (69d0a3e)

### network
#### biosynthesis
##### Changed 🔄
- `plot_network` now builds its interactive plot (hover tooltips, pan, zoom) as self-contained HTML instead of via `bokeh`, and `draw_glycans = True` figures also get hover tooltips in Jupyter (81a910f)
- `construct_network` and `highlight_network` are now faster (eac80ee)
- `get_biosynthetic_coherence` now also takes a filepath or the name of a shipped dataset (ba8b5f7)

##### Fixed 🐛
- Fixed network construction and extension (PTM edge labels, inferred virtual nodes, extension edges and rankings) depending on `PYTHONHASHSEED` (a34092c)
- Fixed `get_differential_biosynthesis`, `construct_network`, and `get_biosynthetic_coherence` using only one row of a glycan listed in several rows (isomer peaks, two spellings of one sequence) instead of their sum (ba8b5f7)
- Fixed `plot_network` crashing on an empty network when drawing glycans (ba8b5f7)

#### evolution
##### Fixed 🐛
- Fixed the member order of `get_communities` communities depending on `PYTHONHASHSEED` (a34092c)
- Fixed `distance_from_embeddings` raising a KeyError for `glycans_to_emb` output paired with a filtered taxonomy frame (ba8b5f7)
- Fixed `dendrogram_from_distance` failing on `pathlib.Path` filepaths and reading the plot format from dots in folder names (ba8b5f7)
