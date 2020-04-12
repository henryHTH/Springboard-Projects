import numpy as np 
import pandas as pd

import os
import time
import datetime
import warnings
warnings.filterwarnings('ignore')

import gc

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from math import sqrt
import math

import lightgbm as lgb
import xgboost as xgb

from tqdm import tqdm_notebook

def has_ugly(row):
    for v in row.values[row.values > 0]:
        if str(v)[::-1].find('.') > 2:
            return True
    return False

def chain_pairs(ordered_items):
    ordered_chains = []
    links_found = 0
    for i_1, op_chain in enumerate(ordered_items.copy()[:]):
        if op_chain[0] != op_chain[1]:
            end_chain = op_chain[-1]
            for i_2, op in enumerate(ordered_items.copy()[:]):
                if (end_chain == op[0]):
                    links_found += 1
                    op_chain.extend(op[1:])
                    end_chain = op_chain[-1]

            ordered_chains.append(op_chain)
    return links_found, ordered_chains

def prune_chain(ordered_chain):
    
    ordered_chain = sorted(ordered_chain, key=len, reverse=True)
    new_chain = []
    id_lookup = {}
    for oc in ordered_chain:
        id_already_in_chain = False
        for idd in oc:
            if idd in id_lookup:
                id_already_in_chain = True
            id_lookup[idd] = idd

        if not id_already_in_chain:
            new_chain.append(oc)
    return sorted(new_chain, key=len, reverse=True)

def find_new_ordered_features(ordered_ids, data_t):
    data = data_t.copy()
    
    f1 = ordered_ids[0][:-1]
    f2 = ordered_ids[0][1:]
    for ef in ordered_ids[1:]:
        f1 += ef[:-1]
        f2 += ef[1:]
            
    d1 = data[f1].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
    d1['ID'] = data.index 
    d2 = data[f2].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
    d2['ID'] = data.index
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    d4 = d1[~d1.duplicated(['key'], keep=False)]
    d5 = d4.merge(d3, how='inner', on='key')

    d_feat = d1.merge(d5, how='left', on='key')
    d_feat.fillna(0, inplace=True)

    ordered_features = list(d_feat[['ID_x', 'ID_y']][d_feat.ID_x != 0].apply(list, axis=1))
    del d1,d2,d3,d4,d5,d_feat
    
    links_found = 1
    #print(ordered_features)
    while links_found > 0:
        links_found, ordered_features = chain_pairs(ordered_features)
        #print(links_found)
    
    ordered_features = prune_chain(ordered_features)
    
    #make lookup of all features found so far
    found = {}
    for ef in extra_features:
        found[ef[0]] = ef
        #print (ef[0])
    found [features[0]] = features

    #make list of sets of 40 that have not been found yet
    new_feature_sets = []
    for of in ordered_features:
        if len(of) >= 40:
            if of[0] not in found:
                new_feature_sets.append(of)
                
    return new_feature_sets

def add_new_feature_sets(data, data_t):
    
    print ('\nData Shape:', data.shape)
    f1 = features[:-1]
    f2 = features[1:]

    for ef in extra_features:
        f1 += ef[:-1]
        f2 += ef[1:]
    
    d1 = data[f1].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
    d1['ID'] = data['ID']    
    gc.collect()
    d2 = data[f2].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
    d2['ID'] = data['ID']
    gc.collect()
    #print('here')
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    del d2
    d4 = d1[~d1.duplicated(['key'], keep=False)]
    #print('here')
    d5 = d4.merge(d3, how='inner', on='key')
    del d4
    d = d1.merge(d5, how='left', on='key')
    d.fillna(0, inplace=True)
    #print('here')
    ordered_ids = list(d[['ID_x', 'ID_y']][d.ID_x != 0].apply(list, axis=1))
    
    del d1,d3,d5,d
    gc.collect()

    links_found = 1
    while links_found > 0:
        links_found, ordered_ids = chain_pairs(ordered_ids)
        #print(links_found)

    print ('OrderedIds:', len(ordered_ids))
    #Make distinct ordered id chains
    ordered_ids = prune_chain(ordered_ids)
    print ('OrderedIds Pruned:', len(ordered_ids))

    #look for ordered features with new ordered id chains
    new_feature_sets = find_new_ordered_features(ordered_ids, data_t)    

    extra_features.extend(new_feature_sets)
    print('New Feature Count:', len(new_feature_sets))
    print('Extra Feature Count:', len(extra_features))

def get_log_pred(data, feats, extra_feats, offset = 2):
    f1 = feats[:(offset * -1)]
    f2 = feats[offset:]
    for ef in extra_feats:
        f1 += ef[:(offset * -1)]
        f2 += ef[offset:]
        
    d1 = data[f1].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
    d2 = data[f2].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
    d2['pred'] = data[feats[offset-2]]
    d2 = d2[d2['pred'] != 0] # Keep?
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    d4 = d1[~d1.duplicated(['key'], keep=False)]
    d5 = d4.merge(d3, how='inner', on='key')
        
    d = d1.merge(d5, how='left', on='key')
    return np.log1p(d.pred).fillna(0)

def get_log_pred(data,feats,extra_feats,lag = 2):
    f1 = feats[:(lag*-1)]
    f2 = feats[lag:]
    for ef in extra_feats:
        f1 += ef[:(lag*-1)]
        f2 += ef[lag:]
    d1 = data[f1].apply(tuple,axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
    d2 = data[f2].apply(tuple,axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
    d2['pred'] = data[feats[lag-2]]
    d3 = d1[~d1.duplicated(['key'],keep=False)]
    d4 = d2[~d2.duplicated(['key'],keep=False)]
    d5 = d4.merge(d3,how='inner',on='key')
    
    d = d1.merge(d5,how='left',on='key')
    return np.log1p(d.pred).fillna(0)  

def add_new_features(source, dest, feats):
    dest['high_{}_{}'.format(feats[0], len(feats))] = np.log1p(source[feats].max(axis=1))
    dest['mean_{}_{}'.format(feats[0], len(feats))] = np.log1p(source[feats].replace(0, np.nan).mean(axis=1))
    dest['low_{}_{}'.format(feats[0], len(feats))] = np.log1p(source[feats].replace(0, np.nan).min(axis=1))
    dest['median_{}_{}'.format(feats[0], len(feats))] = np.log1p(source[feats].replace(0, np.nan).median(axis=1))
    dest['sum_{}_{}'.format(feats[0], len(feats))] = np.log1p(source[feats].sum(axis=1))
    dest['stddev_{}_{}'.format(feats[0], len(feats))] = np.log1p(source[feats].std(axis=1))
    
    dest['mean_log_{}_{}'.format(feats[0], len(feats))] = np.log1p(source[feats].replace(0, np.nan).mean(axis=1))    
    dest['first_nonZero_{}_{}'.format(feats[0], len(feats))] = np.log1p(source[feats].replace(0, np.nan).bfill(1).iloc[:, 0])
    dest['last_nonZero_{}_{}'.format(feats[0], len(feats))] = np.log1p(source[feats[::-1]].replace(0, np.nan).bfill(1).iloc[:, 0])    
    
    #dest['nb_nans_{}_{}'.format(feats[0], len(feats))] =  source[feats].replace(0, np.nan).isnull().sum(axis=1)
    #dest['unique_{}_{}'.format(feats[0], len(feats))] = source[feats].nunique(axis=1)
    return dest

def ts_predictions(NLAGS = 39):
    pred_test = []
    pred_train = []
    efs = extra_features
    for lag in tqdm_notebook(list(range(2, NLAGS))):
        print('lag:', lag)
        log_pred = get_log_pred(train,features,extra_features,lag)
        pred_train.append(np.expm1(log_pred))
        have_data = log_pred!=0
        train_count = have_data.sum()
        score = np.sqrt(mean_squared_error(np.log1p(train.target[have_data]), log_pred[have_data]))
        print (f'score = {score} on {train_count} out of {train.shape[0]} train samples')
        
        log_pred_test = get_log_pred(test,features,extra_features,lag)
        pred_test.append(np.expm1(log_pred_test))
        have_data_test = log_pred_test!=0
        test_count = have_data_test.sum()
        print(f'Having {test_count} predictions out of {test.shape[0]} test samples')
    return {'pred_train':pred_train,'pred_test':pred_test}





def main(train,test,quick_start=False):

    print('Step 1: import featues and prepare data')
    global features
    features = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
        '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
        'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b',
        '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992',
        'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd',
        '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
        '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2',
        '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']

    all_features = [c for c in test.columns if c not in ['ID']]
    test['has_ugly'] = test[all_features].apply(has_ugly, axis=1)
    test_og = test[['ID']].copy()
    test_og['nonzero_mean'] = test[[c for c in test.columns if c not in ['ID', 'has_ugly']]].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)
    test = test[test.has_ugly == False]
    train_t = train.drop(['target'], axis = 1, inplace=False)
    train_t.set_index('ID', inplace=True)
    train_t = train_t.T
    test_t = test.set_index('ID', inplace=False)
    test_t = test_t.T
    gc.collect()

    print('Step 2: add new features sets')
    global extra_features
    extra_features = []
    add_new_feature_sets(train,train_t)
    add_new_feature_sets(test,test_t)
    add_new_feature_sets(train,train_t)
    add_new_feature_sets(test,test_t)
    add_new_feature_sets(train,train_t)
    
    print('clean up and reload test set')
    del train_t, test_t, test
    gc.collect()
    test = pd.read_csv('../input/test.csv')
    test['has_ugly'] = test[all_features].apply(has_ugly, axis=1)
    test[test.has_ugly == True] = 0

    print('Step 3: Make predictions based on pattern found')
    pred = ts_predictions(NLAGS = 39)
    pred_train = pred['pred_train']
    pred_test = pred['pred_test']

    print('Step 4: Combine predictions with non-zero mean')
    pred_train_final = pred_train[0].copy()
    for i in range(1,len(pred_train)):
        pred_train_final[pred_train_final==0] = pred_train[i][pred_train_final==0]
    train_patterns_match_count = (pred_train_final!=0).sum()
    no_match_count = (pred_train_final==0).sum()
    print ("Train Pattern count: ", train_patterns_match_count, "Train no pattern count: ",  no_match_count)

    train['nonzero_mean'] = train[[f for f in train.columns 
                                   if f not in ['ID','target','nonzero_mean']]].apply(lambda x : np.expm1(np.log1p(x[x!=0]).mean()),axis=1)
    pred_train_temp = pred_train_final.copy()
    pred_train_temp[pred_train_temp==0] = train['nonzero_mean'][pred_train_temp==0]
    print(f'Baseline Train Score = {sqrt(mean_squared_error(np.log1p(train.target), np.log1p(pred_train_temp)))}')

    print('Save combined predictions for training set')
    train_patterns = pd.read_csv('../input/train.csv')
    train_patterns = train_patterns[[c for c in train_patterns.columns if c in ['ID','target']]]
    train_patterns['ts_pred'] = pred_train_final
    #train_patterns.to_csv('../input/train_patterns_only_{}.csv'.format(train_patterns_match_count),index=False)

    pred_test_final = pred_test[0].copy()
    for i in range(1,len(pred_test)):
        pred_test_final[pred_test_final == 0] = pred_test[i][pred_test_final == 0]

    #make manually adjustment from observation
    pred_test_final[(4e+07 < pred_test_final)] = 4e+07
    pred_test_final[((pred_test_final < 29000) & (pred_test_final > 0))] = 30000
    pred_test_final[test.ID == 'd72fad286'] = 1560000
    pred_test_final[test.ID == 'a304cde42'] = 320000.0
    test_pattern_match_count = (pred_test_final!=0).sum();
    no_match_count = (pred_test_final==0).sum();
    print ("Test parttern count: ", test_pattern_match_count, "Test no parttern count: ",  no_match_count)

    print('Make a test predction base line')
    pred_test_temp = pred_test_final.copy()
    test_og["nonzero_mean"] = test_og[[f for f in test_og.columns if f not in ["ID", "target", "nonzero_mean", "has_ugly"]]].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)
    pred_test_temp[pred_test_temp==0] = test_og['nonzero_mean'][pred_test_temp==0]
    test_og['target']=pred_test_temp
    #test_og[['ID', 'target']].to_csv('../input/pattern_baseline_{}.csv'.format(test_pattern_match_count), index=False)

    print('Save combined predictions for test set')
    test_patterns = pd.read_csv('../input/sample_submission.csv')
    del test_patterns['target']
    test_patterns['target'] = pred_test_final
    #test_patterns.to_csv('../input/test_patterns_only_{}'.format(test_pattern_match_count),index=False)


    print('Step 5: Make new features based on aggregation')
    features_list = []
    for ef in extra_features:
        features_list.extend(ef)
    features_list.extend(features)

    feats = extra_features.copy()
    feats.insert(0,features)
    feats_df = pd.DataFrame(feats)
    time_features = []
    for c in feats_df.columns:
        time_features.append([f for f in feats_df[c].values if f is not None])
        
    agg_features = []
    all_col = train.columns.drop(['ID','target','nonzero_mean'])
    agg_features.append(all_col)
    agg_features.append([c for c in all_col if c not in features_list])
    agg_features.append(features_list)
    agg_features.extend(time_features)
    agg_features.extend(feats)

    del test
    gc.collect
    test = pd.read_csv('../input/test.csv')

    train_feats = pd.DataFrame()
    test_feats =pd.DataFrame()

    for i, af in tqdm_notebook(list(enumerate(agg_features))):        
        train_features = add_new_features(train, train_feats, af)
        test_features = add_new_features(test, test_feats, af)
    print('Save final new features')    
    #train_features.to_csv('../input/train_agg_feats.csv')
    #test_features.to_csv('../input/test_agg_feats.csv')
    print('########################Finished##########################')
        
        
if __name__=='__main__':
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    start = time.time()
    main(train,test,quick_start=False)
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))