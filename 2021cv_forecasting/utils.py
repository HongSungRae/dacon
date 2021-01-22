import numpy as np
import pandas as pd
import torch
from model import ForecastingCNN


def viewdata(PATH,name):
    df = pd.read_csv(PATH)
    print(">>>> df : {0} , length : {1} <<<<".format(name,len(df)))
    print(df.head())
    return df


def test_dataloader(PATH):
    df = pd.read_csv(PATH)
    x = df.iloc[0:,-1].values
    x = torch.tensor(x).view(1,1,7,48)
    factor = df.iloc[0:,3:7].values
    factor = np.transpose(factor,(1,0))
    factor = torch.tensor(factor).view(1,4,7,48)
    return x,factor


def call_model(PATH):
    model = ForecastingCNN()
    model = torch.load(PATH)
    model.eval()
    return model



if __name__ == "__main__":
    df_train = viewdata('/daintlab/data/sr/dacon/load_forecasting/train/train.csv','train')
    df_test = viewdata('/daintlab/data/sr/dacon/load_forecasting/test/0.csv','test0')
    df_submission = viewdata('/daintlab/data/sr/dacon/load_forecasting/sample_submission.csv','submission')

    print(test_dataloader('/daintlab/data/sr/dacon/load_forecasting/test/0.csv'))