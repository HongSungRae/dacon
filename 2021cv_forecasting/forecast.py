# call up the learned model 
# and put the data to produce the results.
# The results are saved in csv file


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import *


def forecast_target():
    is_cuda =  torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    PATH = '/daintlab/data/sr/dacon/load_forecasting/model/'
    submission = pd.read_csv('/daintlab/data/sr/dacon/load_forecasting/sample_submission.csv')
    print(type(submission))
    for quantile in range(1,10):
        name = str(quantile/10)+'model.pt'
        model = call_model(PATH+name)
        
        for i in range(81): # 0<= i<= 80
            x,factor = test_dataloader('/daintlab/data/sr/dacon/load_forecasting/test/'+str(i)+'.csv')
            if is_cuda:
                x = x.float().cuda()
                factor = factor.float().cuda()
            y_pred = model(x,factor)
            y_pred = y_pred.cpu().view(-1,96)
            y_pred = torch.transpose(y_pred, 0, 1).detach().numpy()
            y_pred = pd.DataFrame(y_pred)
            submission.iloc[i*96:(i+1)*96,quantile] = y_pred.iloc[0:96,0]
            print(type(submission.iloc[i*96:(i+1)*96,quantile]))
        print('>>>>>>> Forecasting Finished When quntile == {} <<<<<<<'.format(quantile/10))
    else:
        submission.to_csv(PATH+'sample_submission.csv')#header, index
        print('      >>>>>>> Every Forecasting Finished <<<<<<<')



if __name__=="__main__":
    forecast_target()