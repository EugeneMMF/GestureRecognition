import torch.nn as nn
import torch
# import tensorflow as tf

# tf.keras.layers.LSTM()

class Model(nn.Module):
    def __init__(self, input_shape:tuple[int,int], lstm_dims:int, classes:int) -> None:
        super(Model, self).__init__()
        self.layers = []
        self.lstm = nn.LSTM(input_size=input_shape[1], hidden_size=lstm_dims, num_layers=1, bias=True, batch_first=True)
        self.dense1 = nn.Linear(in_features=lstm_dims, out_features=lstm_dims, bias=True)
        self.act1 = nn.LeakyReLU(0.05)
        self.dense2 = nn.Linear(in_features=lstm_dims, out_features=classes, bias=True)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x) -> torch.Tensor:
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        return self.softmax(x)