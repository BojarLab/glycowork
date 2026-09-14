# Changelog

## [1.10.1]
- Bumped `glycorender` requirement to `0.4.1` to support `GlycoDraw` die cut stickers (257ba7b)

### glycan_data
#### loader
##### Added ✨
- Added new curated glycomics datasets to `glycomics_data_loader`: `human_influenza_N_10_648982026_02_02_703422`, `human_influenza_O_10_648982026_02_02_703422` (f91d6b4)

### motif
#### tokenization
##### Added ✨
- Added permethylated + peracetylated masses for `HexN`, `PCho`, and `PEtN` (df1cd48)

##### Changed 🔄

##### Fixed 🐛

##### Deprecated ⚠️

#### analysis
##### Changed 🔄
- `annotate_figure` in `get_volcano` is now default-True instead of default-False (f91d6b4)

#### draw
##### Added ✨
- `GlycoDraw` and `display_svg_with_matplotlib` have received the new `sticker` keyword argument, to output die cut sticker style drawings of glycans (257ba7b)

#### annotate
##### Fixed 🐛
- Fixed `get_k_saccharides` outputting disaccharides if `size` > 2 and `up_to=False` ()

### network
#### biosynthesis
##### Changed 🔄
- `draw_glycans` in `plot_network` is now default-True instead of default-False (f91d6b4)