import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas # No, I'm not going to name this pd.
from StockDataset import StockDataset

# Author: Wen Tao (Steven) Zhao
# A Stateful LSTM for predicting small cap stock volume

# Set random seeds to 1: this ensures that the "random" results are the same
torch.manual_seed(1)

# Set device: this is the GPU if that is available, otherwise use the CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
# THESE ARE CURRENTLY NONSENSE
sequence_length = 28
input_size = 2
hidden_size = 128
num_layers = 2
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# Create torch datasets
# NOTE: We need a training set split


training_set = StockDataset('../../Dataset/Microsoft Share Volume Monthly.csv')
# Load torch datasets
training_loader = DataLoader(dataset = training_set, batch_size = batch_size, shuffle = True)



# Simple LSTM Class using Pytorch 0.4.0 for stock volume prediction
class Simple_LSTM(nn.Module):

    # initialize the Class
    def __init__(self, input_size, hidden_size, num_layers):
        # input_size: size of input dimensions.
        # - set to be 2 (timestamp, volume) in testing.
        # hidden_size: Dimensions of the hidden state
        # num_layers: number of LSTM layers
        # num_classes: number of classes in the input
        super(Simple_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        # Note: other activation functions may be used, most notably ReLU
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Set initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Set initial cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate
        out, _ = self.lstm(x, (h0,c0))

        # Decode hidden state from previous time step
        out = self.fc(out[:,-1,:])
        return out

# declare model
model = Simple_LSTM(input_size, hidden_size, num_layers).to(device)


print(model)