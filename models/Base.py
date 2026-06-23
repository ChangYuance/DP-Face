import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from einops import rearrange

from utils import DMIN, CenterLoss


def _make_r3d_18(width_mult=1.0):
    """Build an R3D-18 backbone with optional channel scaling."""
    if abs(width_mult - 1.0) < 1e-6:
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    else:
        model = r3d_18(weights=None)
    if abs(width_mult - 1.0) > 1e-6:
        _scale_channels(model, width_mult)
    return model


def _scale_channels(model, width_mult):
    """Scale channel widths of an R3D-18 in-place."""
    # stem conv: in=3 fixed, out=64*width_mult
    old_stem = model.stem[0]
    new_out = int(64 * width_mult)
    model.stem[0] = nn.Conv3d(
        old_stem.in_channels, new_out,
        kernel_size=old_stem.kernel_size, stride=old_stem.stride,
        padding=old_stem.padding, bias=False)
    model.stem[1] = nn.BatchNorm3d(new_out)

    _scale_blocks(model.layer1, int(64 * width_mult), width_mult)
    _scale_blocks(model.layer2, int(128 * width_mult), width_mult, stride=2)
    _scale_blocks(model.layer3, int(256 * width_mult), width_mult, stride=2)
    _scale_blocks(model.layer4, int(512 * width_mult), width_mult, stride=2)

    model.fc = nn.Linear(int(512 * width_mult), model.fc.out_features)


def _scale_blocks(layer, planes, width_mult, stride=1):
    """Replace BasicBlocks in a layer with width-scaled versions."""
    layer[0] = _scale_basic_block(layer[0], planes, width_mult, stride=stride)
    for i in range(1, len(layer)):
        layer[i] = _scale_basic_block(layer[i], planes, width_mult)


def _scale_basic_block(block, planes, width_mult, stride=1):
    """Build a new BasicBlock with scaled channel widths."""
    inplanes = block.conv1.in_channels
    mid_planes = int(planes * width_mult)
    new_block = nn.Sequential()
    new_block.conv1 = nn.Conv3d(
        inplanes, mid_planes,
        kernel_size=block.conv1.kernel_size, stride=block.conv1.stride,
        padding=block.conv1.padding, bias=False)
    new_block.bn1 = nn.BatchNorm3d(mid_planes)
    new_block.relu = nn.ReLU(inplace=True)
    new_block.conv2 = nn.Conv3d(
        mid_planes, mid_planes,
        kernel_size=block.conv2.kernel_size, stride=stride,
        padding=block.conv2.padding, bias=False)
    new_block.bn2 = nn.BatchNorm3d(mid_planes)

    # downsample if present
    if block.downsample is not None:
        new_block.downsample = nn.Sequential(
            nn.Conv3d(inplanes, mid_planes, kernel_size=1, stride=block.downsample[0].stride, bias=False),
            nn.BatchNorm3d(mid_planes))
    else:
        new_block.downsample = None
    return new_block


class Base(nn.Module):

    def __init__(self, args):
        super(Base, self).__init__()
        self.args = args
        self.device = torch.device(
            'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        self.bag_size = self.args.num_frames // self.args.instance_length
        self.instance_length = self.args.instance_length

        width_mult = getattr(args, 'width_mult', 1.0)
        model = _make_r3d_18(width_mult=width_mult)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = int(512 * width_mult)

        # Freeze encoder except layer3
        for param in self.features.parameters():
            param.requires_grad = False
        if hasattr(self.features[3], 'parameters'):
            for param in self.features[3].parameters():
                param.requires_grad = True

        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=256,
                            num_layers=1, batch_first=True, bidirectional=False)

        self.lstm_dim = 256
        self.heads = 8
        self.dim_head = self.lstm_dim // self.heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(
            self.lstm_dim, (self.dim_head * self.heads) * 3, bias=False)
        self.norm = DMIN(num_features=self.lstm_dim)
        self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)
        self.fc = nn.Linear(self.lstm_dim, self.args.num_classes)
        self.Softmax = nn.Softmax(dim=-1)
        self.center_loss = CenterLoss(num_classes=2, feat_dim=self.lstm_dim, device=self.device, lambda_center=0.5)

    def MIL(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        ori_x = x
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.norm(x)
        x = torch.sigmoid(x)
        x = ori_x * x
        return x

    def forward(self, x):
        x = rearrange(x, 'b (t1 t2) c h w -> (b t1) c t2 h w',
                    t1=self.bag_size, t2=self.instance_length)
        x = self.features(x).squeeze()
        x = rearrange(x, '(b t) c -> b t c', t=self.bag_size)
        x = self.MIL(x)
        x = self.pwconv(x).squeeze()
        out = self.fc(x)
        return out, x
