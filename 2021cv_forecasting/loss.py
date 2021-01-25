import torch
import torch.nn as nn


class Pinball_loss(nn.Module):
    def __init__(self,quantile):
        super().__init__()
        self.quantile = quantile

    def forward(self,y,y_pred):
        a = self.quantile*(y-y_pred)
        b = (1-self.quantile)*(y_pred-y)
        c = torch.max(a,b)
        return torch.sum(c)


if __name__ == "__main__":
    torch.manual_seed(7)
    y = torch.randn(64,1,2,48)
    y_pred = torch.randn(64,1,2,48)
    quantile = 0.5 # 0.1~0.9

    criterion = Pinball_loss(quantile)
    loss = criterion(y,y_pred)
    print('loss :',loss)
    print('type :',type(loss))