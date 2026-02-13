import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from einops import rearrange

from utils import *
class AttentionFusionStable(nn.Module):
    def __init__(self, input_dim_x=1024, input_dim_AUs=128, fused_dim=128, num_heads=4):
        super(AttentionFusionStable, self).__init__()
        self.fused_dim = fused_dim
        self.num_heads = num_heads
        # 卷积映射到相同维度
        self.conv_x = nn.Conv1d(1, fused_dim, kernel_size=3, padding=1)
        self.conv_AUs = nn.Conv1d(1, fused_dim, kernel_size=3, padding=1)
        # 归一化
        self.norm_x = nn.LayerNorm(fused_dim)
        self.norm_AUs = nn.LayerNorm(fused_dim)
        # Multihead Attention
        self.attn = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)
    def forward(self, x, AUs):
        """
        x: [batch, 1024] 视频特征
        AUs: [batch, 128] 面部动作单元特征
        输出: [batch, fused_dim]
        """
        batch_size = x.size(0)
        # ---- 映射 & 归一化 ----
        if batch_size ==1024:
            x = x.unsqueeze(0)
            AUs = AUs.unsqueeze(0)           # [batch, 1, 1024]
        x = x.unsqueeze(1)               # [batch, 1, 1024]
        x = self.conv_x(x)               # [batch, fused_dim, 1024]
        x = x.mean(dim=-1)               # [batch, fused_dim]
        x = self.norm_x(x)
        AUs = AUs.unsqueeze(1)           # [batch, 1, 128]
        AUs = self.conv_AUs(AUs)         # [batch, fused_dim, 128]
        AUs = AUs.mean(dim=-1)           # [batch, fused_dim]
        AUs = self.norm_AUs(AUs)
        # ---- 注意力融合 ----
        # 将 batch 特征扩展为 seq_len=1，适合 MultiheadAttention
        q = x.unsqueeze(1)               # [batch, 1, fused_dim]
        k = AUs.unsqueeze(1)             # [batch, 1, fused_dim]
        v = AUs.unsqueeze(1)             # [batch, 1, fused_dim]
        attn_output, _ = self.attn(q, k, v)  # [batch, 1, fused_dim]
        fused = attn_output.squeeze(1)       # [batch, fused_dim]
        # 可以加残差稳定训练
        fused = fused + x                    # 残差
        return fused

class M3DFEL_AUs_att(nn.Module):
    """The proposed M3DFEL_AUs_att framework
    Args:
        args
    """
    def __init__(self, args):
        super(M3DFEL_AUs_att, self).__init__()
        self.args = args
        self.device = torch.device(
            'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        self.bag_size = self.args.num_frames // self.args.instance_length
        self.instance_length = self.args.instance_length
        # backbone networks
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.features = nn.Sequential(
            *list(model.children())[:-1])  # after avgpool 512x1
        self.lstm = nn.LSTM(input_size=512, hidden_size=512,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.lstm_AUs = nn.LSTM(input_size=20, hidden_size=64,
                            num_layers=2, batch_first=True, bidirectional=True)
        # multi head self attention
        self.heads = 8
        self.dim_head = 1024 // self.heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(
            1024, (self.dim_head * self.heads) * 3, bias=False)
        self.norm = DMIN(num_features=1024)
        # self.to_qkv_AUs = nn.Linear(
        #     1024, (1024 * self.heads) * 3, bias=False)
        # self.norm_AUs = DMIN(num_features=1024)
        self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)
        self.pwconv_AUs = nn.Conv1d(self.args.num_frames, 1, 3, 1, 1)
        # classifier
        self.fc = nn.Linear(128, self.args.num_classes)
        self.Softmax = nn.Softmax(dim=-1)
        self.cross_attention = AttentionFusionStable()
    def check(self, x, name):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"🔥 NaN/Inf in {name}")
            return True
        return False
    def MIL(self, x):
        """The Multi Instance Learning Agregation of instances
        Inputs:
            x: [batch, bag_size, 512]
        """
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        # [batch, bag_size, 1024]
        ori_x = x
        # MHSA
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.norm(x) # torch.Size([8, 4, 1024])
        x = torch.sigmoid(x)
        x = ori_x * x
        return x
    def forward(self, images, AUs):
        # x [batch, 16, 3, 112, 112]
        x = rearrange(images, 'b (t1 t2) c h w -> (b t1) c t2 h w',
                    t1=self.bag_size, t2=self.instance_length)
        # [batch*bag_size, 3, il, 112, 112]
        x = self.features(x).squeeze()
        # [batch*bag_size, 512]
        x = rearrange(x, '(b t) c -> b t c', t=self.bag_size)
        # [batch, bag_size, 20(AUs)]
        x = self.MIL(x)
        AUs[torch.isnan(AUs)] = 0.0
        # [batch, bag_size, 1024]
        if torch.isnan(AUs).any() or torch.isinf(AUs).any():
            print("🔥 NaN/Inf in AU INPUT")
        # [batch, bag_size, 1024]
        x = self.pwconv(x).squeeze()
        AUs = self.pwconv_AUs(AUs).squeeze()
        # [batch, 1024]
        fusion = self.cross_attention(x, AUs)
        out = self.fc(fusion)
        # [batch, 7]
        return out
