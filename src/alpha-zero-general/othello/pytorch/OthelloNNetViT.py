import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import ViT
from torchsummary import summary

class OthelloNNetViT(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(OthelloNNetViT, self).__init__()

        self.vit = ViT(image_size=(self.board_x, self.board_y),
                       channels=1,
                       patch_size=2,
                       num_classes=1024,
                       dim=128,
                       depth=12,
                       heads=16,
                       mlp_dim=1024,
                       dropout=0.1,
                       emb_dropout=0.1)

        self.fc1 = nn.Linear(1024, self.action_size)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, s):
        s = s.unsqueeze(1)
        a = self.vit(s)
        pi = self.fc1(a)  # batch_size x action_size
        v = self.fc2(a)   # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
