import torch
import torch.nn as nn
from einops import rearrange
import models.resnetvgg as ResNet  # 你的 resnetvgg 模块

class FP_VGGFace(nn.Module):
    """Face classification using pretrained VGGFace2 ResNet50 backbone"""

    def __init__(self, args=None):
        super(FP_VGGFace, self).__init__()

        # 1. 初始化 ResNet50 backbone（不包含 top 分类层）
        num_classes = 2  # 输出改为 2 类
        self.backbone = ResNet.resnet50(num_classes=8631, include_top=False)

        # 2. 加载预训练权重
        resume = '/home/chang_yuance/data/changyuance/codes/VGGFace2-pytorch/path/resnet50_ft_weight.pkl'
        import pickle
        with open(resume, 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        self.backbone.load_state_dict(weights)
        print("Pretrained weights loaded successfully!")
        # 3. 新的分类层
        self.fc = nn.Linear(2048, num_classes)  # ResNet50 最后特征维度是 2048

    def forward(self, x):
        """
        x: [batch, 3, 224, 224]
        """
        # 1. 提取特征
        x = x.squeeze(1)
        features = self.backbone(x)  # [batch, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [batch, 2048]

        # 2. 分类输出
        out = self.fc(features)  # [batch, 2]
        return out


# ===================== 测试 =====================
if __name__ == "__main__":
    model = FP_VGGFace()
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)  # batch=2
    with torch.no_grad():
        output = model(dummy_input)
    print("Output shape:", output.shape)  # 应该是 [2, 2]
