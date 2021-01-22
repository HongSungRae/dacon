import torch
import torch.nn as nn
from torchsummary import summary


class ForecastingCNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.cnn_channels_x = CNN_channels()
    self.cnn_channels_factor = CNN_channels(in_channels=4)
    self.linear = nn.Linear(in_features=32000,out_features=96)


  def forward(self,x,factor=torch.zeros([2,4,7,48]).cuda()): # factor = DHI + DNI + WS + RT + T
    x = self.cnn_channels_x(x) # (batch,128,5,25)
    factor = self.cnn_channels_factor(factor) # (batch,128,5,25)
    forecasting = torch.cat([x,factor],dim=1)
    forecasting = forecasting.reshape(forecasting.shape[0],-1) # flatten
    forecasting = self.linear(forecasting)
    return forecasting


class CNN_channels(nn.Module):
  def __init__(self,in_channels=1,dropout=.5,**kwargs):
    super().__init__()

    self.horizontal_channel = Horizontal(in_channels=in_channels,**kwargs)
    self.vertical_channel = Vertical(in_channels=in_channels,**kwargs)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self,x):
    x1 = self.horizontal_channel(x)
    x2 = self.vertical_channel(x)
    x = torch.cat([x1,x2],dim=1)

    return self.dropout(x)


class Horizontal(nn.Module):
  def __init__(self, in_channels = 1):
    super().__init__()

    self.conv1 = conv_block(in_channels=in_channels, out_channels = 16, kernel_size=(1,7))
    self.conv2 = conv_block(16, 24, kernel_size=(1,5))
    self.maxPool1 = nn.MaxPool2d(kernel_size=(1,2),stride=(1,1))
    self.conv3 = conv_block(24, 24, kernel_size =(1,5))
    self.maxPool2 = nn.MaxPool2d(kernel_size=(1,2),stride=(1,1))
    self.conv4 = conv_block(24, 64, kernel_size =(1,4))
    self.maxPool3 = nn.MaxPool2d(kernel_size=(2,1),stride=(1,1))
    self.conv5 = conv_block(64, 64, kernel_size =(1,3))
    self.maxPool4 = nn.MaxPool2d(kernel_size=(2,1),stride=(1,1))
    self.conv6 = conv_block(64, 64, kernel_size =(1,3))

  def forward(self,x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.maxPool1(x)
    x = self.conv3(x)
    x = self.maxPool2(x)
    x = self.conv4(x)
    x = self.maxPool3(x)
    x = self.conv5(x)
    x = self.maxPool4(x)
    x = self.conv6(x)
    return x


class Vertical(nn.Module):
  def __init__(self, in_channels = 1):
    super().__init__()

    self.conv1 = conv_block(in_channels=in_channels, out_channels=16, kernel_size=(4,1),padding=(8,29))
    self.maxPool1 = nn.MaxPool2d(kernel_size=(1,2))
    self.conv2 = conv_block(16, 24, kernel_size=(4,1))
    self.maxPool2 = nn.MaxPool2d(kernel_size=(1,2))
    self.conv3 = conv_block(24, 24, kernel_size=(3,1))
    self.conv4 = conv_block(24, 64, kernel_size=(3,1))
    self.maxPool3 = nn.MaxPool2d(kernel_size=(1,2),stride=(1,1))
    self.conv5 = conv_block(64, 64, kernel_size=(2,1))
    self.maxPool4 = nn.MaxPool2d(kernel_size=(2,1))
    self.conv6 = conv_block(64, 64, kernel_size=(2,1))

  def forward(self,x):
    x = self.conv1(x)
    x = self.maxPool1(x)
    x = self.conv2(x)
    x = self.maxPool2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.maxPool3(x)
    x = self.conv5(x)
    x = self.maxPool4(x)
    x = self.conv6(x)
    return x


class conv_block(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super().__init__()
    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(in_channels,out_channels,**kwargs)

  def forward(self,x):
    return self.relu(self.conv(x))




if __name__ == "__main__":
    net = ForecastingCNN().cuda()
    #net = Vertical().cuda()
    #net = CNN_channels().cuda()
    
    print('======## Parameters of Network ##======')
    summary(net,input_size=(1,7,48))
    x = torch.zeros(64,1,7,48).cuda()
    factor = torch.zeros(64,4,7,48).cuda()
    print(net(x,factor).shape)