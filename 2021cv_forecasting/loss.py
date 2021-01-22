import torch
import torch.nn as nn


class Pinball_loss(nn.Module):
    def __init__(self,quantile):
        super().__init__()
        self.quantile = quantile

    def forward(self,y,y_pred):
        # if *view* is needed, code it here
        return (y-y_pred)*self.quantile if y-y_pred>=0 else (y_pred-y)*(1-self.quantile)


if __name__ == "__main__":
    y = torch.randn(64,1,2,48)
    y_pred = torch.randn(64,1,2,48)
    quantile = 0.5

    criterion = Pinball_loss(quantile)
    loss = criterion(y,y_pred)
    print(loss)