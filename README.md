# Glycowork



![CI](https://github.com/BojarLab/glycowork/workflows/CI/badge.svg)

- Glycans are a fundamental biological sequence, similar to DNA, RNA, or proteins. Glycans are complex carbohydrates that can form branched structures comprising monosaccharides and linkages as constituents. Despite being conspicuously absent from most research, glycans are ubiquitous in biology. They decorate most proteins and lipids and direct the stability and functions of biomolecules, cells, and organisms. This also makes glycans relevant to every human disease.

- The analysis of glycans is made difficult by their nonlinearity and their astounding diversity, given the large number of monosaccharides and types of linkages. Glycowork is a Python package designed to process and analyze glycan sequences, with a special emphasis on glycan-focused machine learning. Next to various functions to work with glycans, Glycowork also contains glycan data that can be used for glycan alignments, model pre-training, motif comparisons, etc.

- The inspiration for glycowork can be found in [Bojar et al., 2020](https://www.cell.com/cell-host-microbe/fulltext/S1931-3128(20)30562-X) and [Burkholz et al., 2021](https://www.biorxiv.org/content/10.1101/2021.03.01.433491v1). There, you can also find examples of possible use cases for the functions in glycowork.

## Install

```python
#later
#`pip install glycowork`

#`pip install git+https://github.com/BojarLab/glycowork.git
#import glycowork`
```

## How to use

Glycowork currently contains four main modules:
 - **alignment**
     - can be used to find similar glycan sequences by alignment according to a glycan-specific substitution matrix
 - **glycan_data**
     - stores several glycan datasets and contains helper functions
 - **ml**
     - here are all the functions for training and using machine learning models, including train-test-split, getting glycan representations, etc.
 - **motif**
     - contains functions for processing glycan sequences, identifying motifs and features, and analyzing them
     
Below are some examples of what you can do with glycowork, be sure to check out the full documentation for everything that's there.

```python
#converting a glycan into a graph object (node list + edge lists)
from glycowork.motif.graph import glycan_to_graph
glycan_to_graph('Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc')
```




    ([802, 1055, 457, 1055, 457, 287, 1025],
     [(0, 1, 2, 3, 5, 6), (1, 2, 3, 4, 6, 4)])



```python
#using graphs, you can easily check whether two glycans are the same - even if they use different bracket notations!
from glycowork.motif.graph import compare_glycans
print(compare_glycans('Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc',
                     'Man(a1-6)[Man(a1-3)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc'))
print(compare_glycans('Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc',
                     'Man(a1-6)[Man(a1-3)]Man(b1-4)GlcNAc(b1-4)GlcNAc'))
```

    True
    False
    

```python
#querying some of the stored databases
from glycowork.motif.query import get_insight
get_insight('Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc')
```

    Let's get rolling! Give us a few moments to crunch some numbers.
    
    This glycan occurs in the following species: ['Antheraea_pernyi', 'Apis_mellifera', 'Autographa_californica_nucleopolyhedrovirus', 'AvianInfluenzaA_Virus', 'Bombyx_mori', 'Bos_taurus', 'Caenorhabditis_elegans', 'Drosophila_melanogaster', 'Homo_sapiens', 'HumanImmunoDeficiency_Virus', 'Mamestra_brassicae', 'Megathura_crenulata', 'Mus_musculus', 'Rattus_norvegicus', 'Spodoptera_frugiperda', 'Sus_scrofa', 'Trichinella_spiralis']
    
    Puh, that's quite a lot! Here are the phyla of those species: ['Arthropoda', 'Chordata', 'Mollusca', 'Negarnaviricota', 'Nematoda', 'Virus']
    
    This glycan contains the following motifs: ['Chitobiose', 'Trimannosylcore', 'core_fucose']
    
    That's all we can do for you at this point!
    

```python
#get motifs, graph features, and sequence features of a set of glycan sequences to train models or analyze glycan properties
glycans = ['Man(a1-3)[Man(a1-6)][Xyl(b1-2)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-3)]GlcNAc',
           'Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-3)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
           'GalNAc(a1-4)GlcNAcA(a1-4)[GlcN(b1-7)]Kdo(a2-5)[Kdo(a2-4)]Kdo(a2-6)GlcOPN(b1-6)GlcOPN']
from glycowork.motif.annotate import annotate_dataset
#HTML(annotate_dataset(glycans, feature_set = ['known', 'graph', 'exhaustive']).head().to_html())
out = annotate_dataset(glycans, feature_set = ['known', 'graph', 'exhaustive']).head()
```

```python
#identify significant binding motifs with (for instance) Z-score data
from glycowork.motif.analysis import get_pvals_motifs
glycans = ['Man(a1-3)[Man(a1-6)][Xyl(b1-2)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-3)]GlcNAc',
           'Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-3)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
           'GalNAc(a1-4)GlcNAcA(a1-4)[GlcN(b1-7)]Kdo(a2-5)[Kdo(a2-4)]Kdo(a2-6)GlcOPN(b1-6)GlcOPN',
           'Man(a1-2)Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
           'Glc(b1-3)Glc(b1-3)Glc']
label = [3.234, 2.423, 0.733, 3.102, 0.108]
test_df = pd.DataFrame({'glycan':glycans, 'binding':label})
out = get_pvals_motifs(test_df, glycan_col_name = 'glycan', label_col_name = 'binding').iloc[:10,:]
```
