import torch.nn as nn
import torch.nn.functional as F


class ConvSDAE(nn.Module):

    def __init__(self, in_channel, out_channel, hidden=256):
        super().__init__()
        self.entry_layer = nn.Conv1d(in_channel, 256, 1)
        self.down1 = nn.Conv1d(256, hidden, 3, padding=1)
        self.hidden_layer = nn.Conv1d(hidden, hidden, 1)
        self.down2 = nn.Conv1d(hidden, 256, 1)
        self.output_layer = nn.Conv1d(256, out_channel, 1)

    def forward(self, x):

        x = F.relu(self.entry_layer(x))
        x = F.relu(self.down1(x))

        x = self.hidden_layer(x)
        x = F.relu(x)
        # x = x.softmax(dim=1)

        x = F.relu(self.down2(x))
        x = self.output_layer(x)

        return F.relu(x)

    def get_hidden_norm(self):
        return self.hidden_layer.weight.norm(p=1)
