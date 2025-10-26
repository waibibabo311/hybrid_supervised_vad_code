import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Memory import *


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True)
        )

    def forward(self, x):

        return self.layer(x)


class Decoder(torch.nn.Module):
    def __init__(self, t_length=2, n_channel=512):
        super(Decoder, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layer(x)


class convAE(torch.nn.Module):
    def __init__(self, n_channel=3, t_length=2, memory_size=10, feature_dim=512, key_dim=512, temp_update=0.1,
                 temp_gather=0.1):
        super(convAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.memory = Memory(memory_size, feature_dim, key_dim, temp_update, temp_gather)

    def forward(self, x, keys, train=True):

        fea = self.encoder(x)
        fea = fea.unsqueeze(-1).unsqueeze(-1)
        if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(
                fea, keys, train)
            updated_fea = updated_fea.squeeze(-1).squeeze(-1)
            output = self.decoder(updated_fea)

            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss

        # test
        else:
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss = self.memory(fea, keys, train)
            updated_fea = updated_fea.squeeze(-1).squeeze(-1)
            output = self.decoder(updated_fea)

            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss





