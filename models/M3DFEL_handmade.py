import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from einops import rearrange
import sys
sys.path.append('/home/chang_yuance/data/changyuance/codes/FCN_train')
from handmade.feature_extract import calculate_2d_features_new
from utils import *

class M3DFEL_handmade(nn.Module):
    """The proposed M3DFEL framework

    Args:
        args
    """

    def __init__(self, args):
        super(M3DFEL_handmade, self).__init__()
        self.args = args
        self.device = torch.device(
            'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        # self.fc_classifier = nn.Linear(29, self.args.num_classes)
        hidden_dim = 64
        self.mlp_classifier = nn.Sequential(
            nn.Linear(29, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # 可选，防止过拟合
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, self.args.num_classes)
        )
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x, Aus):
        hand_made_feature = calculate_2d_features_new(Aus, device=self.device)
        # [batch, 16, 3, 112, 112]
        pooled_feat = torch.max(hand_made_feature, dim=1).values  # [B, 64+29+64]
        # 全连接分类
        out = self.mlp_classifier(pooled_feat)
        # [batch, 7]
        return out
