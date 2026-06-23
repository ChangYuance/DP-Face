import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from einops import rearrange

from utils import DMIN, CenterLoss


class FiLM(nn.Module):
    def __init__(self, input_dim=1024, condition_dim=1024):
        super(FiLM, self).__init__()
        self.fc_gamma = nn.Linear(condition_dim, input_dim)
        self.fc_beta = nn.Linear(condition_dim, input_dim)

    def forward(self, x, cond):
        gamma = self.fc_gamma(cond)
        beta = self.fc_beta(cond)
        return gamma * x + beta


class M3DFEL_AUs_FiLM_center_loss(nn.Module):
    def __init__(self, args):
        super(M3DFEL_AUs_FiLM_center_loss, self).__init__()
        self.args = args
        self.device = torch.device(
            'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        self.bag_size = self.args.num_frames // self.args.instance_length
        self.instance_length = self.args.instance_length

        # backbone
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:-1])

        # image branch
        self.lstm = nn.LSTM(input_size=512, hidden_size=512,
                            num_layers=2, batch_first=True, bidirectional=True)

        # AU branch
        self.lstm_AUs = nn.LSTM(input_size=20, hidden_size=512,
                                num_layers=2, batch_first=True, bidirectional=True)

        # MHSA
        self.heads = 8
        self.dim_head = 1024 // self.heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(1024, (self.dim_head * self.heads) * 3, bias=False)
        self.norm = DMIN(num_features=1024)

        self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)
        self.pwconv_AUs = nn.Conv1d(self.args.num_frames, 1, 3, 1, 1)

        # classifier
        self.fc = nn.Linear(1024, self.args.num_classes)
        self.film = FiLM(input_dim=1024, condition_dim=1024)
        self.center_loss = CenterLoss(num_classes=2, feat_dim=1024, device=self.device, lambda_center=0.5)

    def _ensure_batch_dim(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def MIL(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        ori_x = x
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.norm(x)
        x = torch.sigmoid(x)
        x = ori_x * x
        return x

    def forward(self, images, AUs=None):
        # image feature extraction
        x = rearrange(images, 'b (t1 t2) c h w -> (b t1) c t2 h w',
                      t1=self.bag_size, t2=self.instance_length)
        x = self.features(x).squeeze()
        x = rearrange(x, '(b t) c -> b t c', t=self.bag_size)
        x = self.MIL(x)
        x = self.pwconv(x).squeeze()
        x = self._ensure_batch_dim(x)

        if AUs is not None:
            AUs[torch.isnan(AUs)] = 0.0
            self.lstm_AUs.flatten_parameters()
            AUs, _ = self.lstm_AUs(AUs)
            AUs = self.pwconv_AUs(AUs).squeeze()
            AUs = self._ensure_batch_dim(AUs)
            feature = self.film(x, AUs)
        else:
            feature = x

        out = self.fc(feature)
        return out, feature
