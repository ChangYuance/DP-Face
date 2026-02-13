import torch
import torch.nn as nn
import resnetvgg as ResNet  # 你的 resnetvgg 模块

# 1. 初始化模型（不包含 top 分类层）
num_classes = 8631
model = ResNet.resnet50(num_classes=num_classes, include_top=False)

# 2. 加载预训练权重
resume = '/home/chang_yuance/data/changyuance/codes/VGGFace2-pytorch/path/resnet50_ft_weight.pkl'

import pickle
with open(resume, 'rb') as f:
    obj = f.read()
weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}

# 加载权重到模型
model.load_state_dict(weights)
print("Pretrained weights loaded successfully!")

# 3. 切换到 eval 模式
model.eval()

# 4. 构造一个假输入进行前向测试
# 输入维度: [batch_size, channels, height, width]，VGGFace2 是 224x224 RGB 图
batch_size = 2
dummy_input = torch.randn(batch_size, 3, 224, 224)

# 5. 前向推理
with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)
