import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils import *


class MyDataLoader(Dataset):
    def __init__(self,df):
        super().__init__()
        self.df = df
        self.length = len(self.df) - 9*48 # Last 2days were used for forecasting only.
 
    def __len__(self):
        return self.length # 52128

    def __getitem__(self,idx):
        #print(idx) # 0~63
        start = idx#*48
        x = self.df.iloc[start:start+7*48,-1].values
        y = self.df.iloc[start+7*48:start+9*48,-1].values
        factor = self.df.iloc[start:start+7*48,3:7].values # [336,4]
        factor = np.transpose(factor,(1,0)) # [4,336]

        x = torch.tensor(x).view(1,7,48)
        y = torch.tensor(y).view(1,2,48)
        factor = torch.tensor(factor).view(4,7,48)
        return x, y, factor


if __name__=="__main__":
    df = pd.read_csv('/daintlab/data/sr/dacon/load_forecasting/train/train.csv')
    data_set = MyDataLoader(df)
    data_loader = DataLoader(data_set, shuffle=False, batch_size=64, pin_memory=False)
    x, y, factor = next(iter(data_loader))

    print("length of data loader : {}".format(len(data_loader)))
    print("x : {}".format(x.shape))
    print("y : {}".format(y.shape))
    print("factor : {}".format(factor.shape))