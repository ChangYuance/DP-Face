import torch
import torch.nn as nn
import torchvision.models as models
import math
from marlin_pytorch import Marlin
import sys
sys.path.append('/home/chang_yuance/data/changyuance/codes/AUSTIN/model')
from ASTModel import ASTModel
from s4models import ViS4mer
import torchaudio
import torchaudio.transforms as T

class MAE_only_x(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.mae = Marlin.from_online("marlin_vit_base_ytf")
        self.mae.eval()
        # self.norm = nn.BatchNorm1d(768*2)
        self.decoder = nn.Sequential(
                nn.Linear(768, 512),
                # nn.BatchNorm1d(512),
                nn.ReLU(),
                # nn.Dropout(0.3),
                nn.Linear(512, 2)
            )
    def forward(self, x, phrase='train'):
        """
        Input x is shape (B, L, d_input)
        """
        # x [batch, 16, 3, 224, 224]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.mae.extract_features(x,keep_seq=False) # (B, 768)
        pred_logit = self.decoder(x)  # (B, d_model) -> (B, d_output)
        return pred_logit
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    wav_path = '/home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/audio_test/BaseInfo_1.wav'
    waveform, sample_rate = torchaudio.load(wav_path)
    transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=128)
    mel_spectrogram = transform(waveform)
    if mel_spectrogram.shape[2] < 600:
    # 填充
        mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, 600 - mel_spectrogram.shape[2]))
    elif mel_spectrogram.shape[2] > 600:
        # 裁剪
        mel_spectrogram = mel_spectrogram[:, :, :600]
    print(mel_spectrogram.shape)
    mel_spectrogram = (mel_spectrogram + 4.26) / (4.57 * 2)
    mel_spectrogram = mel_spectrogram.cuda()
    mel_spectrogram = mel_spectrogram.permute(0, 2, 1)
    x1 = torch.rand((2, 3, 16, 224, 224)).cuda()
    # model = ViS4mer(n_layers=3, d_model=3072,d_input=1024,d_output=2,dropout=0.2,l_max=40).cuda()
    model = MAE_AST()
    model = model.cuda()
    print(model)
    model.train()
    o1 = model(x1, None, mel_spectrogram)
    p =1