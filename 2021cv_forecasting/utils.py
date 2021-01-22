import numpy as np
import pandas as pd
import torch


def viewdata(location,name):
    data = pd.read_csv(location)
    print(">>>> data : {0} , length : {1} <<<<".format(name,len(data)))
    print(data.head())



def test_dataloader(location):
    df = pd.read_csv(location)
    x = df.iloc[0:,-1].values
    x = torch.tensor(x).view(1,7,48)
    return x



def call_model(loaction,name):
    
    return model

if __name__ == "__main__":
    viewdata('/daintlab/data/sr/dacon/load_forecasting/train/train.csv','train')
    viewdata('/daintlab/data/sr/dacon/load_forecasting/test/0.csv','test0')

    print(test_dataloader('/daintlab/data/sr/dacon/load_forecasting/test/0.csv'))