import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yaml import load

from sklearn.metrics import accuracy_score, confusion_matrix

from network.functions_output_training import *

# Load configuration file
with open("config.yml") as yaml_config:
    config = load(yaml_config)
    
# Import data
history = pickle.load(open(os.path.join(config['weight_path'], 'train_history_dict'), "rb" ))
test_data = pd.read_csv(config['predictions_output_path'] +'test_data.csv', sep = ';')
predictions = pd.read_csv(config['predictions_output_path'] +'test_predictions.csv', sep = ';')
data = pd.concat([test_data, predictions], axis=1)

## Plot training history
##########################
plot_loss_acc(history, save=False, OutputPath=config['predictions_output_path'])

## Hierarchical classification individual image parts
######################################################

hierarchy = hierarchical_predictions(data)
data_hierarchy = pd.concat([test_data, hierarchy], axis=1)

correct_level = hierarchy.iloc[:,-5:].mean(axis=0)
print(correct_level)

## Hierarchical classification sequences
########################################

# Option 1: Most frequent prediction
######################################

data_seq_hierarchy = freq_predictions(data_hierarchy)

data_seq_hierarchy['level_1_correct'] = data_seq_hierarchy.apply(level_1_correct, axis=1).astype(int)
data_seq_hierarchy['level_2_correct'] = data_seq_hierarchy.apply(level_2_correct, axis=1).astype(int)
data_seq_hierarchy['level_3_correct'] = data_seq_hierarchy.apply(level_3_correct, axis=1).astype(int)
data_seq_hierarchy['level_4_correct'] = data_seq_hierarchy.apply(level_4_correct, axis=1).astype(int)
data_seq_hierarchy['level_5_correct'] = data_seq_hierarchy.apply(level_5_correct, axis=1).astype(int)

correct_level_seq_freq = data_seq_hierarchy.iloc[:,-5:].mean(axis=0)
print(correct_level_seq_freq)

#Convert labels to hierarchical labels
label_hierarchy = data_seq_hierarchy.apply(label_to_hierarchy, axis=1)
label_hierarchy.columns = ['label_level_1','label_level_2','label_level_3','label_level_4','label_level_5']
data_seq_hierarchy = pd.concat([data_seq_hierarchy, label_hierarchy], axis=1)
data_seq_hierarchy['level_5'] = data_seq_hierarchy['level_5'].astype(float)
data_seq_hierarchy['label_level_5'] = data_seq_hierarchy['label_level_5'].astype(float)

# Confusion_matrices
confusion_matrices(data_seq_hierarchy, matrix=False, save=False, OutputPath=config['predictions_output_path'], name='freq_seq')


# Option 2: Top-k prediction
#############################

pred_top = top_predictions(data, data_hierarchy)
pred_top['sequence'] = pred_top['sequence'].astype(str)
data_seq_hierarchy['sequence'] = data_seq_hierarchy['sequence'].astype(str)
pred_top = pred_top.set_index('sequence')
pred_top = data_seq_hierarchy.loc[:,['label','sequence','label_level_1','label_level_2','label_level_3','label_level_4','label_level_5']].join(pred_top, on='sequence', lsuffix='seq')

pred_top['level_1_correct'] = pred_top.apply(level_1_correct, axis=1).astype(int)
pred_top['level_2_correct'] = pred_top.apply(level_2_correct, axis=1).astype(int)
pred_top['level_3_correct'] = pred_top.apply(level_3_correct, axis=1).astype(int)
pred_top['level_4_correct'] = pred_top.apply(level_4_correct, axis=1).astype(int)
pred_top['level_5_correct'] = pred_top.apply(level_5_correct, axis=1).astype(int)

correct_level_seq_top = pred_top.iloc[:,-5:].mean(axis=0)
print(correct_level_seq_top)

# Confusion_matrices
confusion_matrices(pred_top, matrix=False, save=False, OutputPath=config['predictions_output_path'], name='top_seq')

# Final prediction for every sequence
pred_top['prediction_hierarch'] = pred_top.apply(bottom_hierarchical_prediction, axis=1)
cnf_matrix = confusion_matrix(pred_top['label'].values, pred_top['prediction_hierarch'].values)

# Confusion matrix
Names = ['Ass','Marten','Bird','Blank','Cat','Squirrel','Hare','Horse','Human','Mouse','PickupSetup','Fox','Dog','Mouflon','Hedgehog','Deer','Boar']
plot_confusion_matrices_single(cnf_matrix, Names, size=(20,10), norm=True)