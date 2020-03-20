import pandas as pd
import numpy as np
from scipy.stats import spearmanr
def EDA(file_name,data_fraction=None):
    print('version 1')
    if data_fraction is None:
        data = pd.read_csv(file_name)
    else:
        data = pd.read_csv(file_name)
        rows = int(data.shape[0]*data_fraction)
        data = data.sample(n=rows, random_state=1)
    target = data['target']
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
    target = data['target']
    data = data.drop('target',axis = 1)
    df = data.set_index('ID')
    all_zero_l = df_info.columns[np.where(df_info.loc['zero_rate']>=.95)]
    df = df.drop(all_zero_l,axis=1)
    return df,target,df_info