# Changelog

## [1.9.1]

### motif
#### tokenization
##### Added ✨

##### Changed 🔄

##### Fixed 🐛
- Fixed `glycan_to_composition` handling of narrow monosaccharide modification wildcards such as `HexNAc4/6S` (b3ad039)

##### Deprecated ⚠️

#### analysis
##### Added ✨
- `get_biodiversity` now has a new `circadian` API that supports analyzing whether changes in alpha-diversity (internally using `get_jtk`) and beta-diversity, using a distance-based redundancy analysis (db-RDA) regressing the Gower-centered dissimilarity matrix onto a cosinor design, are rhythmic (b4b29c7)

#### processing
##### Fixed 🐛
- Fixed `canonicalize_iupac` messing up narrow modification wildcards (e.g., `Gal3/6S`) in side branches (3a02eff)
- Fixed `canonicalize_iupac` sometimes ordering multiple floating bits wrongly (b6689a5)

#### draw
##### Changed 🔄
- Drawing "forbidden" monosaccharides, such as in `GlycoDraw("Gal(b1-3)[!GlcNAc(b1-6)]GalNAc")`, now automatically makes the forbidden monosaccharides and their linkages transparent (3621350)

#### graph
##### Fixed 🐛
- Fixed `subgraph_isomorphism` counting too many isomorphic matches in certain scenarios of linkage ambiguity (81244c9)
- `get_linkage_number` now always handles narrow linkage wildcards correctly for branch sorting purposes (d222f41)

### network
#### biosynthesis
##### Changed 🔄
- In `construct_network`, `edge_type=enzyme` will now assign glycan class-specific enzymes, if possible (e.g., only ST3GAL4 for N-glycans instead of all ST3GALs)