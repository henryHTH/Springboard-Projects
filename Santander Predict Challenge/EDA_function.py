import pandas as pd
import numpy as np
from scipy.stats import spearmanr
def EDA(file_name):
    data = pd.read_csv(file_name)
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
    return data,df_info