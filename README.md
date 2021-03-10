# Glycowork - A package with helper functions for the processing and analysis of glycan sequences

Contains various helper functions for working with glycan sequences, mostly adapted from [Bojar et al., 2020](https://www.cell.com/cell-host-microbe/fulltext/S1931-3128(20)30562-X) and [Burkholz et al., 2021](https://www.biorxiv.org/content/10.1101/2021.03.01.433491v1).

# Install in Jupyter Notebooks
```
!pip install git+https://YOURUSERNAME:YOURPASSWORD@github.com/BojarLab/glycowork.git
import glycowork
```
Example / Sanity Check:
```
from glycowork.motif.graph import glycan_to_graph
glycan_to_graph('Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc')
```

# To-Do (incomplete)
- add documentation & examples
- fix .csv file imports (**fixed**)
- tests for most of the ML part
- include trained models
