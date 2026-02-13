import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from einops import rearrange
import sys
# sys.path.append('/home/chang_yuance/data/changyuance/codes/3D-ResNets-PyTorch')
from models import resnet
from utils import *


class Palsynet3d(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        pretrained_path = '/home/chang_yuance/data/changyuance/codes/3D-ResNets-PyTorch/pth/r3d50_K_200ep.pth'
        # 1. 构建 R3D-50（按你给的参数写死）
        self.backbone = resnet.generate_model(
            model_depth=50,
            n_classes=700,          # ⚠️ 先保持和预训练一致
            n_input_channels=3,
            shortcut_type='B',
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            widen_factor=1.0
        )

        # 2. 加载预训练权重
        if pretrained_path is not None:
            print(f'Loading pretrained model: {pretrained_path}')
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['state_dict'], strict=True)
        # 3. 替换 fc 为二分类
        self.backbone.fc = nn.Linear(2048, 2)

    def forward(self, x):
        """
        x: [B, 3, T, H, W]
        return: [B, 2]
        """
        x =x.permute(0, 2, 1, 3, 4)  # [B, T, 3, H, W]
        # print(x.shape)
        return self.backbone(x)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Palsynet3d(
        pretrained_path='/home/chang_yuance/data/changyuance/codes/3D-ResNets-PyTorch/pth/r3d50_K_200ep.pth'
    ).to(device)

    model.eval()

    x = torch.randn(2, 3, 16, 112, 112).to(device)

    with torch.no_grad():
        y = model(x)

    print(x.shape)  # [2, 3, 16, 112, 112]
    print(y.shape)  # [2, 2]