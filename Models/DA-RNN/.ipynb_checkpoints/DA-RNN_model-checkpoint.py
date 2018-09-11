"""

Author: Jincheng Zhou

PyTorch implementation of Dual-stage Attention-based LSTM model

Reference: arXiv:1704.02971 [cs, stat]

The model is slightly modified for our project settings

This module only contains the Encoder-decoder Attention-based LSTM model class
No data processing, No training, No evaluation, No anything
Import DARNN from this module to create the DA-RNN model for use

June 23rd, 2018

"""

# Warning: haven't dubugged this yet. Proceed with Caution.


import torch
from torch import nn


# Helper function that checks tensor size

def assertSize(tensor, expected_size, tensor_type_msg = None):
    if tensor_type_msg == None:
        msg = "TensorSizeMismatch: Expect tensor size {}, but received {}." \
            .format(expected_size, tensor.size())
    else:
        msg = "TensorSizeMismatch: Expect {} size {}, but received {}." \
            .format(tensor_type_msg, expected_size, tensor.size())
    assert(tensor.size() == expected_size), msg


# Input Attention Mechanism Module

class InputAttention(nn.Module):
    def __init__(self, num_driving, window_size, en_hidden_size):
        super(InputAttention, self).__init__()
        self.n = num_driving
        self.T = window_size
        self.m = en_hidden_size

        self.FC1 = nn.Linear(2 * self.m + self.T, self.T)
        self.Tanh = nn.Tanh()
        self.FC2 = nn.Linear(self.T, 1)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, X, en_hidden, en_cell, time_step):
        batch_size = X.size()[0]
        # input driving series matrix X is of dimension: (batch_size, n, T)
        # encoder hiden state and cell state vector is of dimension: (batch_size, m)
        assertSize(X, (batch_size, self.n, self.T), "InAttn: Input driving series matrix")
        assertSize(en_hidden, (batch_size, self.m), "InAttn: Encoder hidden state vector")
        assertSize(en_cell, (batch_size, self.m), "InAttn: Encoder cell state vector")

        # Concatenate the driving series to the end of the encoder LSTM's
        #   hidden state and cell state vectors
        en_hidden = en_hidden.unsqueeze(1).expand((batch_size, self.n, self.m))
        en_cell = en_cell.unsqueeze(1).expand((batch_size, self.n, self.m))
        X_concat = torch.cat([en_hidden, en_cell, X], dim=2)
        # Now, X is of dimension: (batch_size, n, 2m+T)
        assertSize(X_concat, (batch_size, self.n, 2 * self.m + self.T), "InAttn: Concatenated driving series matrix")

        # Apply the first affine transformation on the input n driving series
        z = X_concat
        z = self.FC1(z)
        assertSize(z, (batch_size, self.n, self.T), "InAttn: First affine vector")
        # Apply tanh activation function
        z = self.Tanh(z)
        # Apply the second affine transformation
        z = self.FC2(z)
        assertSize(z, (batch_size, self.n, 1), "InAttn: Second affine vector")
        # Apply softmax activation function and squeeze the result tensor
        z = self.Softmax(z)
        z = z.squeeze()
        assertSize(z, (batch_size, self.n), "InAttn: Result weight vector")

        # Compute the weighted extracted driving series at time time_step
        X_t = X[:, :, time_step]
        assertSize(X_t, (batch_size, self.n), "InAttn: Driving series at time_step ")
        X_tild = z * X_t  # tensor element-wise multiplication

        # Return weighted extracted driving series at time time_step
        # X_tild is of dimension: (batch_size, self.n)
        assertSize(X_tild, (batch_size, self.n))

        return X_tild


# Encoder LSTM Module
# Using InputAttention as submodule

class EncoderLSTM(nn.Module):
    def __init__(self, num_driving, window_size, en_hidden_size):
        super(EncoderLSTM, self).__init__()
        self.n = num_driving
        self.T = window_size
        self.m = en_hidden_size

        self.attn = InputAttention(self.n, self.T, self.m)
        self.LSTMCell = nn.LSTMCell(self.n, self.m)

    def forward(self, X):
        batch_size = X.size()[0]
        # input driving series matrix X is of dimension: (batch_size, n, T)
        assertSize(X, (batch_size, self.n, self.T), "Encoder: Input driving series matrix")

        # Initialize hidden state and cell state vectors
        # hiden state and cell state vectors are of dimension: (batch_size, m)
        h, c = [torch.zeros(batch_size, self.m)] * 2
        # Store all historical hidden state vectors
        hidden_states = h.unsqueeze(1)
        assertSize(hidden_states, (batch_size, 1, self.m), "Encoder: Initial hidden states recorder")

        # Forward propagation through timesteps
        for t in range(self.T):
            # Compute the weighted extracted driving series
            X_tild = self.attn(X, h, c, t)
            # Compute the next hidden and cell states through LSTM cell
            h, c = self.LSTMCell(X_tild, (h, c))
            # Store the current hidden state
            hidden_states = torch.cat([hidden_states, h.unsqueeze(1)], dim=1)

        # Discard the first (initial) hidden and cell states in the recorder
        # The recorder matrix is of dimension: (batch_size, T, m)
        hidden_states = hidden_states[:, 1:, :]
        assertSize(hidden_states, (batch_size, self.T, self.m), "Encoder: hidden states matrix")

        # Return the hidden state matrix of length T
        return hidden_states


# Temporal Attention Mechanism Module

class TemporalAttention(nn.Module):
    def __init__(self, window_size, en_hidden_size, de_hidden_size):
        super(TemporalAttention, self).__init__()
        self.T = window_size
        self.m = en_hidden_size
        self.p = de_hidden_size

        self.FC1 = nn.Linear(2 * self.p + self.m, self.m)
        self.Tanh = nn.Tanh()
        self.FC2 = nn.Linear(self.m, 1)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, hidden_states, de_hidden, de_cell, time_step):
        batch_size = hidden_states.size()[0]
        # input encoder hidden states tensor is of dimension: (batch_size, T, m)
        # decoder hidden state and cell state vectors are of dimension: (batch_size, p)
        assertSize(hidden_states, (batch_size, self.T, self.m), "TempAttn: Input encoder hidden states")
        assertSize(de_hidden, (batch_size, self.p), "TempAttn: Decoder hidden states vector")
        assertSize(de_cell, (batch_size, self.p), "TempAttn: Decoder cell states vector")

        # Concatenate the input encoder hidden states to the end of the decoder LSTM's
        #   hidden state and cell state vectors
        de_hidden = de_hidden.unsqueeze(1).expand((batch_size, self.T, self.p))
        de_cell = de_cell.unsqueeze(1).expand((batch_size, self.T, self.p))
        h_concat = torch.cat([de_hidden, de_cell, hidden_states], dim=2)
        # Now, h_concat is of dimension: (batch_size, T, 2p+m)
        assertSize(h_concat, (batch_size, self.T, 2 * self.p + self.m), "TempAttn: Concatenated hidden states matrix")

        # Apply the first affine transformation on the input n driving series
        z = h_concat
        z = self.FC1(z)
        assertSize(z, (batch_size, self.T, self.m), "TempAttn: First affine vector")
        # Apply tanh activation function
        z = self.Tanh(z)
        # Apply the second affine transformation
        z = self.FC2(z)
        assertSize(z, (batch_size, self.T, 1), "TempAttn: Second affine vector")
        # Apply softmax activation function and squeeze the result tensor
        z = self.Softmax(z)

        # Compute the weighted summation of encoder hidden states as the context vector
        C = hidden_states * z  # element-wise multiplication, broadcast on the last dimension
        assertSize(C, (batch_size, self.T, self.m), "TempAttn: weighted input encoder hidden states matrix")
        C = torch.sum(C, 1)  # Summation across T timesteps

        # Return the context vector
        # C is of dimension: (batch_size, m)
        assertSize(C, (batch_size, self.m), "TempAttn: context vector")

        return C


# Decoder LSTM Module
# using TemporalAttention as submodule

class DecoderLSTM(nn.Module):
    def __init__(self, window_size, en_hidden_size, de_hidden_size):
        super(DecoderLSTM, self).__init__()
        self.T = window_size
        self.m = en_hidden_size
        self.p = de_hidden_size

        self.attn = TemporalAttention(self.T, self.m, self.p)
        self.FCin = nn.Linear(self.m + 1, 1)
        self.LSTMCell = nn.LSTMCell(1, self.p)
        self.FCout1 = nn.Linear(self.p + self.m, self.p)
        self.ReLU = nn.ELU()
        self.FCout2 = nn.Linear(self.p, 1)

    def forward(self, hidden_states, y, target_length, teacher_forcing=True):
        batch_size = hidden_states.size()[0]
        # Input encoder hidden states matrix is of dimension: (batch_size, T, m)
        assertSize(hidden_states, (batch_size, self.T, self.m), "Decoder: Input encoder hiden states matrix")
        # target series y is of dimension: (batch_size, target_length)
        # Note:
        ##      - All target series within the current minibatch should have fixed target_length = S
        ##      - The first element of the target series is used to produce the first predicton
        ##        Prediction series and target series both have length S, but shift in one time_step
        ##            i.e. Prediction[t] <==> target[t + 1]
        assertSize(y, (batch_size, target_length), "Decoder: Input target series vectors")

        # Initialize hidden state and cell state vectors
        # hidden state d and cell state s are of dimension: (batch_size, p)
        d, s = [torch.zeros(batch_size, self.p)] * 2

        # Initialize prediction value to the (dummy) first element of the target series
        y_pred = y[:, 0]
        # Initialize history container to store all prediction values across target_length
        # history container is of dimension: (batch_size, target_length)
        y_history = y_pred.unsqueeze(1)
        assertSize(y_history, (batch_size, 1), "Decoder: Initial y_history")

        # Forward propagation through timesteps
        for t in range(target_length):
            # Obtain the context vector at current step
            C = self.attn(hidden_states, d, s, t)

            # Concatenate the context vector to the end of input target value
            if teacher_forcing:
                # If using teacher forcing, extract the input target value from the target series
                y_value = y[:, t]
            else:
                # If not using teacher forcing, use the last prediction value as current
                #   target value
                y_value = y_pred
            # y_value is of dimension: (batch_size, )
            assertSize(y_value, (batch_size,), "Decoder: Current input target value")
            y_value = y_value.unsqueeze(1)
            # Concatenate. After concatenation, y_concat is of dimension: (batch_size, m + 1)
            y_concat = torch.cat([y_value, C], dim=1)
            assertSize(y_concat, (batch_size, self.m + 1), "Decoder: Concatenated input series")

            # Apply the first affine transformation on input series to get the input for LSTM
            LSTM_in = self.FCin(y_concat)

            # Compute the next hidden and cell state through the LSTM unit
            d, s = self.LSTMCell(LSTM_in, (d, s))

            # Concatenate the hidden state vector to the end of context vector
            # The concatenated vector is of dimension: (batch_size, p + m)
            FC_in = torch.cat([d, C], dim=1)
            assertSize(FC_in, (batch_size, self.p + self.m), "Decoder: Concatenated hidden and context vector")

            # Apply the second affine transformation on the concatenated output of LSTM unit
            # Output is of dimension: (batch_size, p)
            z = self.FCout1(FC_in)
            assertSize(z, (batch_size, self.p), "Decoder: Output of first affine transform after LSTM")
            # Apply the ReLU activation function (Note: This step was not in the paper!)
            z = self.ReLU(z)
            # Apply the third affine transformation
            # output is of dimension: (batch_size, 1)
            z = self.FCout2(z)
            assertSize(z, (batch_size, 1), "Decoder: Output of second affine transform after LSTM")

            # Final Prediction
            y_pred = z.squeeze()
            # Add prediction value to history
            y_history = torch.cat([y_history, z], dim=1)

        # Discard the first element in the history of prediction values.
        y_history = y_history[:, 1:]
        # history is of dimension: (batch_size, target_length)
        assertSize(y_history, (batch_size, target_length), "Decoder: Final prediction value series")

        return y_history


# the entire DA-RNN, a wrapper class combining encoder and decoder

class DARNN(nn.Module):
    def __init__(self, num_driving, window_size, en_hidden, de_hidden):
        super(DARNN, self).__init__()
        self.n = num_driving
        self.T = window_size
        self.m = en_hidden
        self.p = de_hidden

        self.encoder = EncoderLSTM(self.n, self.T, self.m)
        self.decoder = DecoderLSTM(self.T, self.m, self.p)

    def forward(self, X, y, target_length, teacher_forcing=True):
        # Forward through encoder
        hidden_states = self.encoder(X)
        # Forward through decoder
        y_history = self.decoder(hidden_states, y, target_length, teacher_forcing)

        # Return the final prediction
        return y_history

