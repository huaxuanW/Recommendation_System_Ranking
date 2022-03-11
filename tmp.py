import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from model import DeepFM
from feature_name import *
from tqdm import tqdm
from dataset import MyDataset1

df = pd.read_csv("dealer_prioritization_upstream_open_new.csv")

df['purchase_date'] = pd.to_datetime(df['purchase_date']) 

train_idx = df.iloc[:602121,:].index
train = df[train_idx]
df_neg = train[train['label']==0]
df_pos = train[train['label']==1]
df_neg_1 = df_neg.sample(n=len(df_pos) * 2, random_state = 2022, replace = False)

train_idx = pd.concat([df_neg_1,df_pos], axis = 0).index
test_idx = df.iloc[602121:,:].index


for feat in tqdm(categorical_feature):
    lbe = LabelEncoder()
    df[feat] = lbe.fit_transform(df[feat])

scaler = MinMaxScaler()
df[numerical_feature] = scaler.fit_transform(df[numerical_feature])

train = df.iloc[train_idx]
test = df.iloc[test_idx]

lstm_train = pd.read_csv('lstm_dataset_train.csv', names= lstm_feature)
lstm_test = pd.read_csv('lstm_dataset_test.csv', names= lstm_feature)

train = pd.concat([train, lstm_train], axis=1)
test = pd.concat([test, lstm_test], axis=1)

train_dataset = MyDataset1(train, numerical_feature, categorical_feature, lstm_feature, target)
test_dataset = MyDataset1(test, numerical_feature, categorical_feature, lstm_feature, target)

