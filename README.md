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
- adapt annotate_dataset to deal with wildcards; approximate matching
- tests for some of the ML part
- include trained models

# Minimum example to train a model
```
!pip install torch==1.7.0
import torch

!pip install torch-geometric \
  torch-sparse==latest+cu102 \
  torch-scatter==latest+cu102 \
  torch-cluster==latest+cu102 \
  torch-spline-conv==latest+cu102 \
  -f https://pytorch-geometric.com/whl/torch-1.7.0.html
  
!pip install git+https://YOURUSERNAME:YOURPASSWORD@github.com/BojarLab/glycowork.git
import glycowork

from glycowork.glycan_data.loader import df_species, lib
from glycowork.ml.train_test_split import hierarchy_filter
from glycowork.ml.processing import split_data_to_train
from glycowork.ml import model_training

train_x, val_x, train_y, val_y, id_val, class_list, class_converter = hierarchy_filter(df_species,
                                                                                       rank = 'kingdom')
dataloaders = split_data_to_train(train_x, val_x, train_y, val_y)

model = model_training.prep_model('SweetNet', len(class_list))
optimizer_ft = torch.optim.Adam(model.parameters(), lr = 0.0005, weight_decay = 0.001)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, 50)
criterion = torch.nn.CrossEntropyLoss().cuda()
model_ft = model_training.train_model(model, dataloaders, criterion, optimizer_ft, scheduler,
                   num_epochs = 100)
```
