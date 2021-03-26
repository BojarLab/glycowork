import glycowork

from glycowork.glycan_data.loader import df_species
from glycowork.ml.train_test_split import *

train_x, val_x, train_y, val_y, id_val, class_list, class_converter = hierarchy_filter(df_species,
                                                                                       rank = 'Kingdom')
print(train_x[:10])

train_x, val_x, train_y, val_y, id_val, class_list, class_converter = hierarchy_filter(df_species,
                                                                                       rank = 'Kingdom',
                                                                                       wildcard_seed = True,
                                                                                       wildcard_list = linkages,
                                                                                       wildcard_name = 'bond')
print(train_x[-10:])
