import numpy as np
import pandas as pd
from utils import *


def viewdata(location,name):
    data = pd.read_csv(location)
    print(">>>> data : {0} , length : {1} <<<<".format(name,len(data)))
    print(data.head())


if __name__ == "__main__":
    viewdata('/daintlab/data/sr/dacon/load_forecasting/train/train.csv','train')
    viewdata('/daintlab/data/sr/dacon/load_forecasting/test/0.csv','test0')