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

#### draw
##### Changed 🔄
- Drawing "forbidden" monosaccharides, such as in `GlycoDraw("Gal(b1-3)[!GlcNAc(b1-6)]GalNAc")`, now automatically makes the forbidden monosaccharides and their linkages transparent