import numpy as np
import pandas as pd
import time
import torch
import torch.optim as optim
from model import ForecastingCNN
from loss import Pinball_loss
from dataloader import MyDataLoader
from utils import *


def train(model,data_loader,quantile,epochs):
    print('>>>>>>> Learning Start! quantile : {} <<<<<<<'.format(quantile))
    start = time.time()

    is_cuda =  torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    loss_list = []

    net = model
    optimizer = opim.SGD(net.parameters())
    criterion = Pinball_loss(quantile)
    total_batch = len(data_loader)
    print('total_batch : {}'.format(total_batch))

    for eph in range(epochs):
        loss_train = 0.0
        print('epoch / epochs = {} / {}'.format(eph+1,epochs))

        for i,data in enumerate(data_loader):
            x, y, factor = data
            if is_cuda:
                x = x.cuda()
                y = y.cuda()
                factor = factor.cuda()

            optimizer.zero_grad()
            y_hat = net(x,factor)
            loss = criterion(y,y_hat)
            loss_train += loss
            loss.backward()
            optimizer.step()

            if i%100 == 99:
                loss_train = 0.0
                print('epoch : {} , iter : {} , loss : {}'
                        .format(eph,i+1,loss_train.item()/100))
            if i == total_batch-1:
                loss_list.append(loss_train.item()/(total_batch%100))
        
    end = time.time()
    print('>>>>>>> Learning Finished! Time taken : {} <<<<<<<<'.format(end-start)))
    PATH = '/daintlab/data/sr/dacon/load_forecasting/model'
    NAME = str(quantile) + 'model.pt'
    torch.save(net, PATH + NAME)

    return loss_list


if __nmae__ == "__main__":
    global_start = time.time()
    loss = []

    df = pd.read_csv('/daintlab/data/sr/dacon/load_forecasting/train/train.csv')
    data_set = MyDataLoader(df)
    data_loader = DataLoader(data_set, shuffle=False, batch_size=64, pin_memory=False)


    ######################
    ## hyper parameters ##
    _quantile = 0.0 # DO NOT modify
    epochs = 10
    ######################
    ######################


    for _ in range(1,10):
        _quantile += 0.1
        loss_list = train(model, data_loader,_quantile,epochs)
        loss.append(loss_list)
    else:
        global_end = time.time()
        print("\n\n")
        print("###############################################")
        print(loss)
        print("###############################################")
        print(">>>>>>> All training were compeleted!!! <<<<<<<")
        print(">>>>>>> It took {} seconds <<<<<<<".format(global_start-global_end))