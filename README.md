# Glycowork



## Install

```python
#later
#`pip install glycowork`

#if you don't have a security token
#`pip install git+https://YOURUSERNAME:YOURPASSWORD@github.com/BojarLab/glycowork.git

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
     - contains functions for processing glycan sequences, identifying motifs and featues, and analyzing them
     
Below are some examples of what you can do with glycowork, be sure to check out the full documentation for everything that's there.

```python
#converting a glycan into a graph object (node list + edge lists)
from glycowork.motif.graph import glycan_to_graph
glycan_to_graph('Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc')
```




    ([794, 1045, 450, 1045, 450, 281, 1015],
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
    
    This glycan contains the following motifs: ['Chitobiose', 'Trimannosylcore', 'core_fucose']
    
    That's all we can do for you at this point!
    

```python
#get motifs, graph features, and sequence features of a set of glycan sequences to train models or analyze glycan properties
glycans = ['Man(a1-3)[Man(a1-6)][Xyl(b1-2)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-3)]GlcNAc',
           'Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-3)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
           'GalNAc(a1-4)GlcNAcA(a1-4)[GlcN(b1-7)]Kdo(a2-5)[Kdo(a2-4)]Kdo(a2-6)GlcOPN(b1-6)GlcOPN']
from glycowork.motif.annotate import annotate_dataset
display(annotate_dataset(glycans, feature_set = ['known', 'graph', 'exhaustive']))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LewisX</th>
      <th>LewisY</th>
      <th>SialylLewisX</th>
      <th>SulfoSialylLewisX</th>
      <th>LewisA</th>
      <th>LewisB</th>
      <th>SialylLewisA</th>
      <th>SulfoLewisA</th>
      <th>H_type2</th>
      <th>H_type1</th>
      <th>...</th>
      <th>GlcNAcA*a1-4*Kdo</th>
      <th>GlcOPN*b1-6*GlcOPN</th>
      <th>Kdo*a2-4*Kdo</th>
      <th>Kdo*a2-5*Kdo</th>
      <th>Kdo*a2-6*GlcOPN</th>
      <th>Man*a1-2*Man</th>
      <th>Man*a1-3*Man</th>
      <th>Man*a1-6*Man</th>
      <th>Man*b1-4*GlcNAc</th>
      <th>Xyl*b1-2*Man</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Man(a1-3)[Man(a1-6)][Xyl(b1-2)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-3)]GlcNAc</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-3)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>GalNAc(a1-4)GlcNAcA(a1-4)[GlcN(b1-7)]Kdo(a2-5)[Kdo(a2-4)]Kdo(a2-6)GlcOPN(b1-6)GlcOPN</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 1236 columns</p>
</div>


```python
#identify significant binding motifs with (for instance) Z-score data
import pandas as pd
from glycowork.motif.analysis import get_pvals_motifs
glycans = ['Man(a1-3)[Man(a1-6)][Xyl(b1-2)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-3)]GlcNAc',
           'Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-3)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
           'GalNAc(a1-4)GlcNAcA(a1-4)[GlcN(b1-7)]Kdo(a2-5)[Kdo(a2-4)]Kdo(a2-6)GlcOPN(b1-6)GlcOPN',
           'Man(a1-2)Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
           'Glc(b1-3)Glc(b1-3)Glc']
label = [3.234, 2.423, 0.733, 3.102, 0.108]
test_df = pd.DataFrame({'glycan':glycans, 'binding':label})
display(get_pvals_motifs(test_df, glycan_col_name = 'glycan', label_col_name = 'binding'))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>motif</th>
      <th>pval</th>
      <th>corr_pval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1075</th>
      <td>Man*b1-4*GlcNAc</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1074</th>
      <td>Man*a1-6*Man</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1073</th>
      <td>Man*a1-3*Man</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>450</th>
      <td>GlcNAc</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1015</th>
      <td>a1-6</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>364</th>
      <td>GalNFoAN</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>365</th>
      <td>GalNOPCho</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>366</th>
      <td>GalNSuc</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>352</th>
      <td>GalNAcOAcOMeA</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1076</th>
      <td>Xyl*b1-2*Man</td>
      <td>0.211325</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>1077 rows × 3 columns</p>
</div>

