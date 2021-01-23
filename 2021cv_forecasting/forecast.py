# call up the learned model 
# and put the data to produce the results.
# The results are saved in csv file


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import *


def forecast_target():
    PATH = '/daintlab/data/sr/dacon/load_forecasting/model/'
    submission = pd.read_csv('/daintlab/data/sr/dacon/load_forecasting/sample_submission.csv')
    for quantile in range(1,10):
        name = str(quantile/10)+'model.pt'
        model = call_model(PATH+name)
        
        for i in range(81): # 0<= i<= 80
            x,factor = test_dataloader('/daintlab/data/sr/dacon/load_forecasting/test/'+str(i)+'.csv')
            y_pred = model(x,factor)
            y_pred = y_pred.view(-1,96)
            submission.iloc[i*48:(i+2)*48,quantile] = y_pred
        else:
            print('>>>>>>> Forecasting Finished <<<<<<<')
    else:
        print('>>>>>>> Every Forecasting Finished <<<<<<<')



if __name__=="__main__":
    forecast_target()