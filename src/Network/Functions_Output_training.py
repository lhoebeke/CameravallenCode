import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
from collections import Counter

######################################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=10, fontweight='bold', labelpad=20)
    plt.xlabel('Predicted label', fontsize=10, fontweight='bold', labelpad=20)

####################################################################
def plot_confusion_matrices(cnf_matrix, class_names, size = (24,12)):
    
    """
    This function plots both the confusion matrix and the normalized confusion matrix.
    """
    
    plt.figure(figsize=size)
    plt.subplot(121) 
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.subplot(122)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=1, hspace=None)

####################################################################
def plot_confusion_matrices_single(cnf_matrix, class_names, size = (12,6), norm=False):
    
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    plt.figure(figsize=size)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=norm,
                          title='Normalized confusion matrix')
    
###################################################################
def plot_loss_acc(history, save= False, OutputPath=None):
    
    """
    This function plots the training and validation loss and accuracy.
    Normalization can be applied by setting `normalize=True`.
    """
    
    plt.figure(figsize=(12,12))
    plt.subplot(211)  
    plt.plot(history['categorical_accuracy']) 
    plt.plot(history['val_categorical_accuracy']) 
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['categorical_accuracy', 'val_categorical_accuracy' ], loc='upper left') 

    plt.subplot(212)  
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train','validation'], loc='upper left')  
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=None, hspace=1)
    
    if save:
        plt.savefig(os.path.join(OutputPath, 'Loss_Acc.png'))

#######################################################################
"""
These functions are applied to hierarchically classify the images.

level_n(x): determines the prediction at level n. If there is no prediction to be made at level n, the function returns nan.
level_n_p(x): determines the probability associated with the prediction at level n. This probability is needed to classify the sequences using the top-k method.
level_n_correct: determines if the prediction at level n is correct.
"""
    
def level_1(x):
    if x['blank'] > x['not_blank']:
        return 'blank'
    else:
        return 'not_blank'

def level_1_p(x):
    if x['blank'] > x['not_blank']:
        return x['blank']
    else:
        return x['not_blank']

def level_2(x):
    if x['level_1'] == 'not_blank':
        if x['animal'] > x['no_animal']:
            return 'animal'
        else:
            return 'no_animal'
    else:
        return math.nan
 
def level_2_p(x):
    if x['animal'] > x['no_animal']:
        return x['animal']
    else:
        return x['no_animal']

def level_3(x):
    if x['level_2'] == 'animal':
        if x['bird'] > x['mammal']:
            return 'bird'
        else:
            return 'mammal'
    elif x['level_2'] == 'no_animal':
        if x['human'] > x['pickup']:
            return 'human'
        else:
            return 'pickup'        
    else:
        return math.nan
 
def level_3_p(x):
    if x['level_2'] == 'animal':
        if x['bird'] > x['mammal']:
            return x['bird']
        else:
            return x['mammal']
    elif x['level_2'] == 'no_animal':
        if x['human'] > x['pickup']:
            return x['human']
        else:
            return x['pickup']      
    else:
        return math.nan
 
def level_4(x):
    if x['level_3'] == 'mammal':
        if x['small_mammal'] > x['large_mammal']:
            return 'small_mammal'
        else:
            return 'large_mammal'      
    else:
        return math.nan

def level_4_p(x):
    if x['level_3'] == 'mammal':
        if x['small_mammal'] > x['large_mammal']:
            return x['small_mammal']
        else:
            return x['large_mammal']    
    else:
        return math.nan

def level_5(x,pred):
    if x['level_4'] == 'small_mammal':
        p = int(pred.iloc[x.name,[9,5,6,14]].idxmax())
        return p
    elif x['level_4'] == 'large_mammal':
        p = int(pred.iloc[x.name,[0,7,11,1,4,12,13,15,16]].idxmax())
        return p
    else:
        return math.nan

def level_5_p(x,pred):
    if x['level_4'] == 'small_mammal':
        p = np.asarray(pred.iloc[x.name,[9,5,6,14]]).max(axis=0)
        return p
    elif x['level_4'] == 'large_mammal':
        p = np.asarray(pred.iloc[x.name,[0,7,11,1,4,12,13,15,16]]).max(axis=0)
        return p
    else:
        return math.nan

def level_1_correct(x):
    if x['level_1'] == 'blank':
        return x['label'] == 3
    else:
        return x['label'] != 3

def level_2_correct(x):
    if x['level_2'] == 'animal':
        return x['label'] in [2,9,5,6,14,0,7,11,1,4,12,13,15,16]
    elif x['level_2'] == 'no_animal':
        return x['label'] in [8,10]
    else:
        return x['level_1_correct']
    
def level_3_correct(x):
    if x['level_3'] == 'bird':
        return x['label'] == 2
    elif x['level_3'] == 'mammal':
        return x['label'] in [9,5,6,14,0,7,11,1,4,12,13,15,16]
    elif x['level_3'] == 'human':
        return x['label'] == 8
    elif x['level_3'] == 'pickup':
        return x['label'] == 10
    else:
        return x['level_2_correct']
    
def level_4_correct(x):
    if x['level_4'] == 'small_mammal':
        return x['label'] in [9,5,6,14]
    elif x['level_4'] == 'large_mammal':
        return x['label'] in [0,7,11,1,4,12,13,15,16]
    else:
        return x['level_3_correct']

def level_5_correct(x):
    if math.isnan(x['level_5']):
        return x['level_4_correct']
    else:
        return x['level_5'] == x['label']
    
######################################################################
def label_to_hierarchy(x):
    
    """
    This function converts the output labels to the label at every level of the classification tree.
    """
    
    if x['label'] == 0:
        return pd.Series(['not_blank', 'animal','mammal','large_mammal', 0])
    elif x['label'] == 1:
        return pd.Series(['not_blank', 'animal','mammal','large_mammal', 1])
    elif x['label'] == 2:
        return pd.Series(['not_blank', 'animal','bird', math.nan, math.nan])
    elif x['label'] == 3:
        return pd.Series(['blank', math.nan,math.nan,math.nan, math.nan])
    elif x['label'] == 4:
        return pd.Series(['not_blank', 'animal','mammal','large_mammal', 4])
    elif x['label'] == 5:
        return pd.Series(['not_blank', 'animal','mammal','small_mammal', 5])
    elif x['label'] == 6:
        return pd.Series(['not_blank', 'animal','mammal','small_mammal', 6])
    elif x['label'] == 7:
        return pd.Series(['not_blank', 'animal','mammal','large_mammal', 7])
    elif x['label'] == 8:
        return pd.Series(['not_blank', 'no_animal','human', math.nan, math.nan])
    elif x['label'] == 9:
        return pd.Series(['not_blank', 'animal','mammal','small_mammal', 9])
    elif x['label'] == 10:
        return pd.Series(['not_blank', 'no_animal','pickup',math.nan, math.nan])
    elif x['label'] == 11:
        return pd.Series(['not_blank', 'animal','mammal','large_mammal', 11])
    elif x['label'] == 12:
        return pd.Series(['not_blank', 'animal','mammal','large_mammal', 12])
    elif x['label'] == 13:
        return pd.Series(['not_blank', 'animal','mammal','large_mammal', 13])
    elif x['label'] == 14:
        return pd.Series(['not_blank', 'animal','mammal','small_mammal', 14])
    elif x['label'] == 15:
        return pd.Series(['not_blank', 'animal','mammal','large_mammal', 15])
    elif x['label'] == 16:
        return pd.Series(['not_blank', 'animal','mammal','large_mammal', 16])
    
########################################################################
def top_predictions(data, data_hierarchy):
    
    """
    This function determines the prediction of the sequences based on the top-k predictions at every level.
    """
    
    sequences = data_hierarchy['sequence'].drop_duplicates()
    data_seq_top = pd.DataFrame(columns=['sequence','level_1','level_1_p','level_2','level_2_p','level_3','level_3_p','level_4','level_4_p','level_5','level_5_p'])
    
    for s, seq in enumerate(sequences):
        data_sequence = data_hierarchy[data_hierarchy['sequence'] == seq]
        pred_sequence = data.loc[:,'0':'16'].loc[data['sequence'] == seq]
        
        #Level 1
        p_l1 = max([data_sequence['blank'].mean(),  data_sequence['not_blank'].mean()])
        l1 = ['blank', 'not_blank'][np.argmax([data_sequence['blank'].mean(),  data_sequence['not_blank'].mean()])]
        
        #level 2
        if l1 == 'not_blank':
            p_l2 = max([data_sequence['animal'].mean(),  data_sequence['no_animal'].mean()])
            l2 = ['animal', 'no_animal'][np.argmax([data_sequence['animal'].mean(),  data_sequence['no_animal'].mean()])]
        else:
            p_l2 = math.nan
            l2 = math.nan
        
        #Level 3
        if l2 == 'animal':
            p_l3 = max([data_sequence['bird'].mean(),  data_sequence['mammal'].mean()])
            l3 = ['bird', 'mammal'][np.argmax([data_sequence['bird'].mean(),  data_sequence['mammal'].mean()])]
        elif l2 == 'no_animal':
            p_l3 = max([data_sequence['human'].mean(),  data_sequence['pickup'].mean()])
            l3 = ['human', 'pickup'][np.argmax([data_sequence['human'].mean(),  data_sequence['pickup'].mean()])]
        else:
            p_l3 = math.nan
            l3 = math.nan     
            
        #Level 4
        if l3 == 'mammal':
            p_l4 = max([data_sequence['small_mammal'].mean(),  data_sequence['large_mammal'].mean()])
            l4 = ['small_mammal', 'large_mammal'][np.argmax([data_sequence['small_mammal'].mean(),  data_sequence['large_mammal'].mean()])]
        else:
            p_l4 = math.nan
            l4 = math.nan 
    
        #Level 5
        if l4 == 'small_mammal':
            p_l5 = max(pred_sequence.iloc[:,[9,5,6,14]].mean())
            l5 = int(np.argmax(pred_sequence.iloc[:,[9,5,6,14]].mean()))
        elif l4 == 'large_mammal':
            large = pred_sequence.iloc[:,[0,7,11,1,4,12,13,15,16]]
            
            top5_p = []
            top5_pred = []
            #Top-5 for every image
            for i, row in large.iterrows():
                top5_p += np.sort(row.values.tolist())[-5:].tolist()
                top5_pred += np.array([0,7,11,1,4,12,13,15,16])[np.argsort(row.values.tolist())[-5:].tolist()].tolist()
            df_top5 = pd.DataFrame({'top5_p': top5_p, 'top5_pred':top5_pred})
            top5_seq = df_top5.groupby('top5_pred').sum().divide(len(data_sequence)).sort_values('top5_p',ascending=False)[:5]
            
            p_l5 = top5_seq.max()[0]
            l5 = int(top5_seq.idxmax()[0])
            
        else:
            p_l5 = math.nan
            l5 = math.nan 
        
        data_seq_top.loc[s] = [seq, l1, p_l1, l2, p_l2, l3, p_l3, l4, p_l4, l5, p_l5] 
    
    return data_seq_top

#########################################################################################
def hierarchical_predictions(data):

    """
    This function determines the hierarchical prediction for the individual images, based on the output of the neural network.
    These predictions can then be used to classify a sequence using the frequency method.
    """
    
    predictions = data.loc[:,'0':'16']
    
    index_small = [9,5,6,14]
    index_large = [0,7,11,1,4,12,13,15,16]
    
    hierarchy = pd.DataFrame()
    hierarchy['blank'] = predictions.iloc[:,3]
    hierarchy['small_mammal'] = predictions.iloc[:,index_small].sum(axis=1)
    hierarchy['large_mammal'] = predictions.iloc[:,index_large].sum(axis=1)
    hierarchy['mammal'] = hierarchy['small_mammal'] + hierarchy['large_mammal'] 
    hierarchy['bird'] = predictions.iloc[:,2]
    hierarchy['animal'] = hierarchy['bird']  + hierarchy['mammal']
    hierarchy['human'] = predictions.iloc[:,8]
    hierarchy['pickup'] = predictions.iloc[:,10]
    hierarchy['no_animal'] = hierarchy['human'] + hierarchy['pickup'] 
    hierarchy['not_blank'] = hierarchy['no_animal'] + hierarchy['animal']
        
    hierarchy['level_1'] = hierarchy.apply(level_1, axis=1)
    hierarchy['level_1_p'] = hierarchy.apply(level_1_p, axis=1)
    hierarchy['level_2'] = hierarchy.apply(level_2, axis=1)
    hierarchy['level_2_p'] = hierarchy.apply(level_2_p, axis=1)
    hierarchy['level_3'] = hierarchy.apply(level_3, axis=1)
    hierarchy['level_3_p'] = hierarchy.apply(level_3_p, axis=1)
    hierarchy['level_4'] = hierarchy.apply(level_4, axis=1)
    hierarchy['level_4_p'] = hierarchy.apply(level_4_p, axis=1)
    
    mammals = pd.DataFrame()
    mammals['small_pred_max'] = np.asarray(predictions.iloc[:,index_small]).argmax(axis=1)
    mammals['large_pred_max'] = np.asarray(predictions.iloc[:,index_large]).argmax(axis=1)
    mammals['small_max_p'] = np.asarray(predictions.iloc[:,index_small]).max(axis=1)
    mammals['large_max_p'] = np.asarray(predictions.iloc[:,index_large]).max(axis=1)
    
    hierarchy['level_5'] = hierarchy.apply(level_5, pred = predictions, axis=1)
    hierarchy['level_5_p'] = hierarchy.apply(level_5_p,  pred = predictions, axis=1)
    hierarchy['label'] = data['label']
    
    hierarchy['level_1_correct'] = hierarchy.apply(level_1_correct, axis=1).astype(int)
    hierarchy['level_2_correct'] = hierarchy.apply(level_2_correct, axis=1).astype(int)
    hierarchy['level_3_correct'] = hierarchy.apply(level_3_correct, axis=1).astype(int)
    hierarchy['level_4_correct'] = hierarchy.apply(level_4_correct, axis=1).astype(int)
    hierarchy['level_5_correct'] = hierarchy.apply(level_5_correct, axis=1).astype(int)
    
    return hierarchy

############################################################################################
def confusion_matrices(prediction, matrix=False, save=False, OutputPath=None, name=None):
    
    """
    This function plot the confusion matrix at every level of the classification tree.
    """    
    
    cnf_matrix_level_1 = confusion_matrix(prediction['label_level_1'].values,prediction['level_1'].values, labels=['blank', 'not_blank'])
    cnf_matrix_level_2 = confusion_matrix(prediction['label_level_2'].values.astype('str'),prediction['level_2'].values.astype('str'), labels=['animal', 'no_animal',math.nan])
    cnf_matrix_level_3 = confusion_matrix(prediction['label_level_3'].values.astype('str'),prediction['level_3'].values.astype('str'), labels=['bird', 'mammal', 'human', 'pickup', math.nan])
    cnf_matrix_level_4 = confusion_matrix(prediction['label_level_4'].values.astype('str'),prediction['level_4'].values.astype('str'), labels=['small_mammal', 'large_mammal', math.nan])
    cnf_matrix_level_5 = confusion_matrix(prediction['label_level_5'].values.astype('str'),prediction['level_5'].values.astype('float32').astype('str'), labels=['9.0','5.0','6.0','14.0', '0.0','7.0','11.0','1.0','4.0','12.0','13.0','15.0','16.0',math.nan])
    
    plot_confusion_matrices(cnf_matrix_level_1, ['Blank', 'Not blank'], size=(8,4))
    fig1 = plt.gcf()
    plot_confusion_matrices(cnf_matrix_level_2, ['Animal', 'No animal',math.nan])
    fig2 = plt.gcf()
    plot_confusion_matrices(cnf_matrix_level_3, ['Bird', 'Mammal', 'Human', 'PickupSetup', math.nan])
    fig3 = plt.gcf()
    plot_confusion_matrices(cnf_matrix_level_4, ['small mammal', 'Large mammal', math.nan])
    fig4 = plt.gcf()
    plot_confusion_matrices(cnf_matrix_level_5, ['Mouse','squirrel','hare','hedgehog','ass','horse','fox','marten','cat', 'dog','mouflon','deer','boar', math.nan])
    fig5 = plt.gcf()
    
    if save:
        fig1.savefig(os.path.join(OutputPath, 'Confusion_'+name+'_level_1.pdf'), format='pdf', dpi=1000, bbox_inches="tight")
        fig2.savefig(os.path.join(OutputPath, 'Confusion_'+name+'_level_2.pdf'), format='pdf', dpi=1000, bbox_inches="tight")
        fig3.savefig(os.path.join(OutputPath, 'Confusion_'+name+'_level_3.pdf'), format='pdf', dpi=1000, bbox_inches="tight")
        fig4.savefig(os.path.join(OutputPath, 'Confusion_'+name+'_level_4.pdf'), format='pdf', dpi=1000, bbox_inches="tight")
        fig5.savefig(os.path.join(OutputPath, 'Confusion_'+name+'_level_5.pdf'), format='pdf', dpi=1000, bbox_inches="tight")
    
    if matrix:
        return cnf_matrix_level_1, cnf_matrix_level_2, cnf_matrix_level_3, cnf_matrix_level_4, cnf_matrix_level_5

###############################################################################################
def freq_predictions(data_hierarchy):
    
    """
    This function determines the prediction of the sequences by taking the most frequent prediction over the images.
    """
    
    seq = data_hierarchy[['label', 'sequence']].drop_duplicates().iloc[:,1:]
    pred_seq_hierarchy = pd.DataFrame()
    
    column_names_seq = ['level_1_seq', 'level_1_p_seq','level_2_seq', 'level_2_p_seq','level_3_seq', 'level_3_p_seq','level_4_seq', 'level_4_p_seq','level_5_seq', 'level_5_p_seq']
    for n, name in enumerate(column_names_seq):
        pred_seq_hierarchy[name] = data_hierarchy[name[:-4]].groupby(data_hierarchy['sequence']).apply(list)
    pred_seq_hierarchy.reset_index(inplace=True)
    
    column_names = ['level_1','level_2','level_3','level_4','level_5']
    for name in column_names:
        pred_seq_hierarchy[name] = ""
        for i, row in pred_seq_hierarchy.iterrows():
            count = Counter(row.loc[name+'_seq'])
            pred_seq_hierarchy[name].iloc[i] = count.most_common()[0][0]
    
    pred_seq_hierarchy['sequence'] = pred_seq_hierarchy['sequence'].astype(str)
    pred_seq_hierarchy = pred_seq_hierarchy.set_index('sequence')
    data_seq_hierarchy = seq.join(pred_seq_hierarchy, on='sequence', lsuffix='seq')
    
    return data_seq_hierarchy

############################################################################################
def bottom_hierarchical_prediction(x):
    
    """
    This function determines the final prediction for a sequence, based on the hierarchical prediction at every level.
    """
    
    if pd.isnull(x['level_5']) == False:
        label = x['level_5']
    elif pd.isnull(x['level_3']) == False:
        label = x['level_3']
        if label == 'bird':
            label = 2
        elif label == 'human':
            label = 8
        else:
            label = 10
    else:
        label = 3 #blank

    return label
