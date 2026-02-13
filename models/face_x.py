import torch
import torch.nn as nn
import sys
sys.path.append('/home/chang_yuance/data/changyuance/codes/AUSTIN/model')
from facexformer import FaceXFormer  # 修改为你实际 import 路径
class FacialFeatureProcessor2(nn.Module):
    def __init__(self, llm_embedding_dim=512):
        super().__init__()
        # 每个尺度 feature map 输出通道数
        self.channels = [128, 256, 512, 1024]
        hidden_dim = 128  # 大幅降低 hidden_dim
        self.fusion_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for c in self.channels
        ])
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim*len(self.channels), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, llm_embedding_dim)
        )

    def forward(self, multi_scale_features):
        fused = []
        for feat, module in zip(multi_scale_features, self.fusion_modules):
            # GAP -> [B, C]
            x = feat.mean(dim=[2,3])
            x = module(x)
            fused.append(x)
        x = torch.cat(fused, dim=1)
        x = self.projector(x)
        return x
class FacialFeatureProcessor(nn.Module):
    def __init__(self, llm_embedding_dim=768):
        """
        Args:
            llm_embedding_dim (int): Final output dimension matching LLM embedding size
        """
        super().__init__()
        # Calculate flattened dimensions for each scale
        # [1, 128, 56, 56] -> 128 * 56 * 56
        # [1, 256, 28, 28] -> 256 * 28 * 28
        # [1, 512, 14, 14] -> 512 * 14 * 14
        # [1, 1024, 7, 7] -> 1024 * 7 * 7
        self.flat_dims = [
            128 * 56 * 56,    # 401,408
            256 * 28 * 28,    # 200,704
            512 * 14 * 14,    # 100,352
            1024 * 7 * 7      # 50,176
        ]
        hidden_dim = 512  # Intermediate dimension for fusion
        # MLP Fusion modules for each scale
        self.fusion_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for flat_dim in self.flat_dims
        ])
        # Calculate total dimension after concatenation
        total_fusion_dim = hidden_dim * len(self.flat_dims)
        # Facial projector (two linear layers)
        self.projector = nn.Sequential(
            nn.Linear(total_fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, llm_embedding_dim)
        )
    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features (list): List of feature maps with shapes:
                                    [1, 128, 56, 56]
                                    [1, 256, 28, 28]
                                    [1, 512, 14, 14]
                                    [1, 1024, 7, 7]
        Returns:
            torch.Tensor: Projected features matching LLM embedding dimension
                        [batch_size, llm_embedding_dim]
        """
        # Flatten spatial dimensions for each scale
        flattened_features = []
        for feat in multi_scale_features:
            # Reshape from [B, C, H, W] to [B, C*H*W]
            flat = feat.flatten(1)
            flattened_features.append(flat)
        # Apply MLP fusion to each scale
        fused_features = []
        for feat, fusion_module in zip(flattened_features, self.fusion_modules):
            fused = fusion_module(feat)
            fused_features.append(fused)
        # Concatenate all fused features
        concat_features = torch.cat(fused_features, dim=1)
        # Project to LLM embedding dimension
        projected_features = self.projector(concat_features)
        return projected_features
class face_x(nn.Module):
    def __init__(self, llm_embedding_dim=512):
        """
        输入视频帧 [B, L, 3, 224, 224] -> 输出二分类 logits [B, 2]
        """
        super().__init__()
        # 初始化 FaceXFormer
        self.encoder = FaceXFormer()
        self.encoder.eval()  # 特征提取模式
        for p in self.encoder.parameters():
            p.requires_grad = False
        # 特征处理器，将多尺度特征映射到固定维度
        self.processor = FacialFeatureProcessor2(llm_embedding_dim=llm_embedding_dim)
        # 二分类 decoder
        self.decoder = nn.Sequential(
            nn.Linear(llm_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.lstm = nn.LSTM(input_size=llm_embedding_dim, hidden_size=llm_embedding_dim,
                    num_layers=2, batch_first=True, bidirectional=True)
    def forward(self, x):
        """
        x: [B, L, 3, 224, 224]
        """
        B, L = x.shape[0], x.shape[1]
        # reshape 输入给 FaceXFormer
        x_reshaped = x.view(B*L, 3, 224, 224)
        # 模拟 label/task 输入，保证 forward 不报错
        labels = None
        tasks = torch.ones(B*L, dtype=torch.long, device=x.device)
        # 提取多尺度特征
        with torch.no_grad():
            self.encoder(x_reshaped, labels, tasks)
            multi_scale_features = [f.to(x.device) for f in self.encoder.multi_scale_features]
        # 特征处理器
        feat = self.processor(multi_scale_features)  # [B*L, llm_embedding_dim]
        # reshape 回 [B, L, d_model] 并做 temporal pooling（max）
        feat = feat.view(B, L, -1)
        feat = self.lstm(feat)[0]
        # decoder 输出二分类
        logits = self.decoder(feat.max(dim=1)[0])  # [B, 2]
        return logits
if __name__ == "__main__":
    model = facex_x().cuda()
    x = torch.randn(4, 16, 3, 224, 224).cuda()  # batch 4
    logits = model(x)
    print("Logits shape:", logits.shape)  # [4, 2]

