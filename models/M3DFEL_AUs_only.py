import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from einops import rearrange

from utils import *


class M3DFEL_AUs_only(nn.Module):
    """The proposed M3DFEL_AUs framework
    Args:
        args
    """
    def __init__(self, args):
        super(M3DFEL_AUs_only, self).__init__()
        self.args = args
        self.device = torch.device(
            'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        self.bag_size = self.args.num_frames // self.args.instance_length
        self.instance_length = self.args.instance_length
        # backbone networks
        self.lstm_AUs = nn.LSTM(input_size=20, hidden_size=64,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.pwconv_AUs = nn.Conv1d(self.args.num_frames, 1, 3, 1, 1)
        # classifier
        self.fc = nn.Linear(128, self.args.num_classes)
        self.Softmax = nn.Softmax(dim=-1)
    def forward(self, images, AUs):
        # [batch, bag_size, 1024]
        AUs[torch.isnan(AUs)] = 0.0
        AUs, _  = self.lstm_AUs(AUs) # B T 20
        # [batch, bag_size, 1024]
        AUs = self.pwconv_AUs(AUs).squeeze()
        out = self.fc(AUs)
        # [batch, 7]
        return out
