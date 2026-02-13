import torch
import torch.nn as nn
 
class FiLM(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(FiLM, self).__init__()
        
        # 全连接层，用于生成γ和β参数
        self.fc_gamma = nn.Linear(condition_dim, input_dim)
        self.fc_beta = nn.Linear(condition_dim, input_dim)
        
    def forward(self, x, condition):
        # 根据条件特征获取缩放scale参数和移位参数shift，即计算γ和β参数
        gamma = self.fc_gamma(condition)
        beta = self.fc_beta(condition)
        
        # 对输入特征x进行缩放和偏移，实现条件特征调整输入特征
        y = gamma * x + beta 
        return y
 
if __name__ == "__main__":
    input_dim = 64 # 输入特征
    condition_dim = 128 # 条件特征
 
    # 创建一个FiLM层实例
    film_layer = FiLM(input_dim, condition_dim)
 
    # 初始化输入特征x和条件特征condition
    x = torch.randn(1, input_dim)
    condition = torch.randn(1, condition_dim)
 
    # 使用FiLM层对输入特征x进行条件调整
    y = film_layer(x, condition)
 
    print(y.shape) # [1, 64]