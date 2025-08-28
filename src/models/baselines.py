import torch
import torch.nn as nn

class CNNBaseline(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: [batch, features] -> [batch, 1, features] for Conv1d
        x = x.unsqueeze(1)
        return self.net(x)

class RNNBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, features] -> [batch, seq_len, input_dim]
        x = x.unsqueeze(1)  # seq_len=1
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out
class TransformerBaseline(nn.Module):
    def __init__(self, input_dim, num_heads, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [batch, features] -> [batch, seq_len, input_dim]
        x = x.unsqueeze(1)  # seq_len=1
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])
        return x
