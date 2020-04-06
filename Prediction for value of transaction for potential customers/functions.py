import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn_gbmi import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,make_scorer
from sklearn.ensemble import RandomForestRegressor 
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,KFold 
from EDA_function import EDA
from pprint import pprint
from math import sqrt

def EDA(file_name,data_fraction=None,zero_level=0.95):
    print('version 5')
    if data_fraction is None:
        data = pd.read_csv(file_name)
    else:
        data = pd.read_csv(file_name)
        rows = int(data.shape[0]*data_fraction)
        data = data.sample(n=rows, random_state=1)
    target = np.log1p(data['target'].values)
    df = data.iloc[:,2:]
    zero_rate = []
    target_cor = []
    col_var = []
    col_unique_num = []
    rank_corr = []
    rank_corr_p = []
    col_missing_value = df.isnull().sum().sum() # the number is 0, which means no missing value
    col_var = df.var(axis=0)

    for col in df.columns:
        col_unique_num.append(df[col].nunique())
        target_cor.append(data[['target',col]].corr().values[0][1])
        zero_rate.append(len(df[col].iloc[np.where(df[col]==0)])/len(df[col]))
        coef,p = spearmanr(data.target,df[col])
        rank_corr.append(coef)
        rank_corr_p.append(p)
    df_info = {'type':df.dtypes,'var':col_var,'unique_number':col_unique_num,'corr':target_cor,'zero_rate':zero_rate,'spearman_corr':rank_corr,'spearman_corr_p':rank_corr_p}
    df_info = pd.DataFrame(df_info).T
    data = data.drop('target',axis = 1)
    df = data.set_index('ID')
    all_zero_l = df_info.columns[np.where(df_info.loc['zero_rate']>=zero_level)]
    df = df.drop(all_zero_l,axis=1)
    return df,target,df_info
    #if data_fraction is None:
    #    return df,target,df_info
    #else:
    #    rows = int(df.shape[0]*data_fraction)
    #    return df.sample(n=rows, random_state=1),target.sample(n=rows, random_state=1),df_info

def neg_rmse(y,y_pred):

    return -np.sqrt(mean_squared_error(y,y_pred))

def random_grid(X_train, y_train,model,param_grid,RANDOM_SEED=42,n_iter =20,cv=5,score_method = neg_rmse):
    rf_random = RandomizedSearchCV(estimator = model, param_distributions = param_grid, scoring = make_scorer(neg_rmse,greater_is_better=True),
                              n_iter =n_iter, cv =cv, verbose=2, random_state=RANDOM_SEED, n_jobs = 8,return_train_score=True)
    rf_random.fit(X_train, y_train)
    return rf_random

def search_grid(X_train, y_train,model,param_grid,n_iter =20,cv=5):
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, scoring = make_scorer(neg_rmse,greater_is_better=True),
                              cv = 5, n_jobs = 8, verbose = 2, return_train_score=True)
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    test_results = (model.score(X_test, y_test),sqrt(mean_squared_error(predictions, y_test)))
    print('Model Performance')
    print('R2 Score: {:0.2f}% degrees.'.format(test_results[0]))
    print('RMSE = {:0.4f}'.format(test_results[1]))
    return test_results

def feature_sel(model,X_train,y_train):
    sel = SelectFromModel(model)
    sel.fit(X_train, y_train)
    selected_feat= X_train.columns[(sel.get_support())]
    #len(selected_feat)
    print(selected_feat)
    return selected_feat


def plot_results(model, param = 'n_estimators', name = 'Num Trees'):
    param_name = 'param_%s' % param

    # Extract information from the cross validation model
    train_scores = model.cv_results_['mean_train_score']
    test_scores = model.cv_results_['mean_test_score']
    train_time = model.cv_results_['mean_fit_time']
    param_values = list(model.cv_results_[param_name])
    
    # Plot the scores over the parameter
    plt.subplots(1, 2, figsize=(10, 6))
    plt.subplot(121)
    plt.plot(param_values, train_scores, 'bo-', label = 'train')
    plt.plot(param_values, test_scores, 'go-', label = 'test')
    #plt.ylim(ymin = -10, ymax = 0)
    plt.legend()
    plt.xlabel(name)
    plt.ylabel('neg_root_mean_squared_error')
    plt.title('Score vs %s' % name)
    
    plt.subplot(122)
    plt.plot(param_values, train_time, 'ro-')
    #plt.ylim(ymin = 0.0, ymax = 2.0)
    plt.xlabel(name)
    plt.ylabel('Train Time (sec)')
    plt.title('Training Time vs %s' % name)
      
    plt.tight_layout(pad = 4)

def permutation_importance(df,train_idx,test_idx):
    #scores = defaultdict(list)
    scores = {}
    #rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
    #for train_idx, test_idx in rs.split(df):
    print("TRAIN:", train_idx, "TEST:", test_idx)
    X_train, X_test = df.iloc[train_idx,:], df.iloc[test_idx,:]
    Y_train, Y_test = target[train_idx], target[test_idx]
    r = rf_best.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf_best.predict(X_test))
    for col in df:
        X_t = X_test.copy()
        np.random.shuffle(X_t.loc[:,col])
        shuff_acc = r2_score(Y_test, rf_best.predict(X_t))
        #scores[col].append((acc-shuff_acc)/acc)
        scores[col] = ((acc-shuff_acc)/acc)
    print ("Features sorted by their score:")
    #sorted_scores= sorted([(round(np.mean(score), 4), feat) for
    #              feat, score in scores.items()], reverse=True)
    return scores 

def get_all_permutation_importance(df):
    """
    We just recursively fetch target value for different lags
    """
    df_copy =  df.copy()
    CPU_CORES = 8
    rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
    with Pool(processes=CPU_CORES) as p:
        res = [p.apply_async(permutation_importance, args=(df_copy,train_idx,test_idx)) for train_idx, test_idx in rs.split(df)]
        res = [r.get() for r in res]
    
    #for i in range(nlags):
    #    print("Processing lag {}".format(i))
    #    df["leaked_target_"+str(i)] = _get_leak(df, cols, i)
    pool.close()
    scores_list = defaultdict(list)
    for i in range(5):
        for score,val in scores[i].items():
            scores_list[score].append(val)
    scores_list
    sorted_scores= sorted([(round(np.mean(score), 4), feat) for
                    feat, score in scores_list.items()], reverse=True)
    return sorted_scores

def cross_validate(X,y,xgb,rf,features,xgb_fit_params):
    loss = []
    kf = KFold(n_splits=5, random_state=137, shuffle=False)
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        xgb.fit(X_train[features], y_train,
                eval_set=[(X_train[features], y_train), (X_test[features], y_test)],
                **xgb_fit_params)
        rf.fit(X_train,y_train)
        xgb_pred = xgb.predict(X_test[features])
        rf_pred = rf.predict(X_test)
        final_pred = 0.2 * xgb_pred + 0.8 * rf_pred
        loss.append(xgb_para['loss_func'](final_pred, y_test))
    return loss

def main(data_fraction=None):

    RANDOM_SEED = 137
    test_fraction = 0.2
    print('Loading and EDA')
    df,target,df_info = EDA('train.csv',DataFrame_fraction)

    param_grid = grid_param_setup()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size= test_fraction, random_state=RANDOM_SEED)
    base_model = RandomForestRegressor(random_state = RANDOM_SEED)
    rf_random = random_grid(X_train,y_train,base_model,param_grid,RANDOM_SEED)
    #rf_grid = search_grid(X_train,y_train,base_model,param_grid_search) 

    #compare_result(rf_random,rf_grid,base_model,X_train, X_test, y_train, y_test)
    return rf_random
    #return rf_random, rf_grid


if __name__ == '__main__':
    import warnings
    import time
    warnings.filterwarnings('ignore')
    start = time.time()
    
    main()


    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))




    '''
    {'bootstrap': True,
    'max_depth': 120,
    'max_features': 'sqrt',
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'n_estimators': 120}
    '''