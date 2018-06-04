import numpy as np
import math
import pandas as pd

#######################################################################
"""
These functions are applied to hierarchically classify the images.

level_n(x): determines the prediction at level n. If there is no prediction to be made at level n, the function returns nan.
level_n_p(x): determines the probability associated with the prediction at level n. This probability is needed to classify the sequences using the top-k method.
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
    These predictions can then be used to classify a sequence.
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
    
    return hierarchy

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
    
    labels = ['Ass','Beech Marten','Bird','Blank','Cat','Squirrel','Hare','Horse','Human','Mouse','PickupSetup','Fox','Dog','Mouflon','Hedgehog','Roe Deer','Wild Boar']
    label = labels[label]

    return label
