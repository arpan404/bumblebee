"""
This file contains the code for neural network models used in the project. They are copied from the training notebooks.
"""

import torch
import torch.nn as nn

"""
Attention and CursorRNN are copied from 'rnn-train.ipynb' notebook
Attention class is used to compute the attention weights and context vector
CursorRNN class is the main model that combines LSTM, attention, and residual connections to predict the path
"""


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(
            hidden_dim, 1, bias=False
        )  # Attention layer to assign weights to different time steps

    def forward(self, lstm_out):
        scores = self.attn(lstm_out)  # Compute attention scores for each time step
        attn_weights = torch.softmax(
            scores, dim=1
        )  # Apply softmax to normalize attention weights
        context = torch.sum(
            attn_weights * lstm_out, dim=1
        )  # Create context vector by weighted sum of LSTM outputs

        return context, attn_weights


class CursorRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.2):
        super(CursorRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )  # LSTM layer for sequence processing
        self.attention = Attention(
            hidden_dim
        )  # Attention mechanism to focus on important time steps
        self.residual_fc = nn.Linear(
            input_dim, hidden_dim
        )  # Residual connection to help with gradient flow
        self.layer_norm = nn.LayerNorm(
            hidden_dim
        )  # Layer normalization for training stability
        self.fc = nn.Linear(
            hidden_dim, output_dim
        )  # Output projection layer to generate final predictions

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Process sequence through LSTM
        context, attn_weights = self.attention(
            lstm_out
        )  # Apply attention to focus on relevant parts
        residual = self.residual_fc(
            x[:, -1, :]
        )  # Create residual connection from last input
        combined = self.layer_norm(
            context + residual
        )  # Combine attention output with residual and normalize
        output = self.fc(combined)  # Generate final trajectory prediction

        return output, attn_weights
