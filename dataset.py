from torch.utils.data import Dataset
import pandas as pd
import torch


class MyDataset(Dataset):
    def __init__(self, filepath):
        super().__init__()

        df = pd.read_csv(filepath)
        x = df.iloc[:,:90].values
        y = df.iloc[:,90].values

        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class MyDataset1(Dataset):
    def __init__(self, df, num_fea, cate_fea, hist_fea, label):
        super().__init__()
        self.num_fea = torch.FloatTensor(df.loc[:, num_fea].values)
        self.cate_fea = torch.LongTensor(df.loc[:, cate_fea].values)
        self.hist_fea = torch.FloatTensor(df.loc[:, hist_fea].values)
        self.label = torch.LongTensor(df.loc[:, label].values)
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return self.num_fea[index], self.cate_fea[index], self.hist_fea[index], self.label[index]