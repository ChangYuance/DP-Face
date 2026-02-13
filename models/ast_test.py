import os 
import torch
import sys
sys.path.append('/home/chang_yuance/data/changyuance/codes/AUSTIN/model')
from ASTModel import ASTModel
# download pretrained model in this directory
os.environ['TORCH_HOME'] = '/home/chang_yuance/data/changyuance/codes/AUSTIN/model/pretrained_models'
# assume each input spectrogram has 100 time frames
input_tdim = 100
# assume the task has 527 classes
label_dim = 527
# create a pseudo input: a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins 
test_input = torch.rand([10, input_tdim, 128]) 
# create an AST model
ast_mdl = ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=True)
test_output = ast_mdl(test_input)
# output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes. 
print(test_output.shape)