import pandas as pd
from sklearn import preprocessing
from feature_name import *
import numpy as np
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

def get_data():
    """
    return dnn_feature_columns, linear_feature_columns, train_model_input, test_model_input
    """
    df = pd.read_csv('/content/drive/MyDrive/深度学习/data/dealer_prioritization_upstream_open_new.csv')

    #fillna
    df[categorical_feature] = df[categorical_feature].fillna('-1')
    df[numerical_feature] = df[numerical_feature].fillna(0)

    #categorical feature

    for feat in categorical_feature:
        lbe = preprocessing.LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])

    #numerical feature

    scaler = preprocessing.MinMaxScaler()
    df[numerical_feature] = scaler.fit_transform(df[numerical_feature])

    #target

    df[target] = df[target].astype(np.int32)

    #get nunique and other stuff
    fixlen_feature_columns = [SparseFeat(feat, df[feat].nunique()+1) for feat in categorical_feature] + [DenseFeat(feat, 1, ) for feat in numerical_feature]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
    linear_feature_columns + dnn_feature_columns)

    df1 = pd.read_csv('/content/drive/MyDrive/深度学习/data/dealer_prioritization_upstream_open_new.csv')

    train_idx = df1[df1['purchase_date'] <= '2021-08-30'].index.tolist()

    test_idx = df1[~(df1['purchase_date'] <= '2021-08-30')].index.tolist()

    #train test split
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    # balance data
    df_neg = train[train[target]==0]
    df_pos = train[train[target]==1]

    df_neg =df_neg.sample(n=len(df_pos) * 3, random_state=2022, replace=False)

    train = pd.concat([df_neg, df_pos], axis=0)


    #generate input data for model
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    return dnn_feature_columns, linear_feature_columns, train_model_input, test_model_input, train[target].values, test[target].values