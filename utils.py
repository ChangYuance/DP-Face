import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
from einops import rearrange
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device, lambda_center=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        self.device = device
        self.lambda_center = lambda_center

    def forward(self, features, labels):
        centers_batch = self.centers[labels]
        center_loss = torch.mean(torch.sum((features - centers_batch) ** 2, dim=1)) / self.feat_dim
        return center_loss

def compute_l2_loss(model, weight_decay):
    l2_loss = 0
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.sum(param ** 2)
    return weight_decay * l2_loss

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets = F.one_hot(targets, self.num_classes)
        p_t = torch.sum(inputs * targets, dim=1)
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)
        return loss.mean()

def get_wrong_path_with_confidence(
    all_pred: np.ndarray,
    all_target: np.ndarray,
    allpath: np.ndarray,
    all_output: torch.Tensor,
    top_k: int = 50,
    use_softmax: bool = True,
    class_names: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
    assert len(all_pred) == len(all_target) == len(allpath) == len(all_output), \
        "Length mismatch in inputs"

    all_pred = np.array(all_pred)
    all_target = np.array(all_target)
    allpath = np.array(allpath)
    all_pred = torch.from_numpy(all_pred)
    all_target = torch.from_numpy(all_target)

    if use_softmax:
        probs = torch.nn.functional.softmax(all_output, dim=1)
    else:
        probs = all_output

    mask_0_to_1 = (all_target == 0) & (all_pred == 1)
    mask_1_to_0 = (all_target == 1) & (all_pred == 0)

    conf_0_to_1_full = probs[:, 1] - probs[:, 0]
    conf_1_to_0_full = probs[:, 0] - probs[:, 1]

    conf_0_to_1_masked = conf_0_to_1_full[mask_0_to_1]
    conf_1_to_0_masked = conf_1_to_0_full[mask_1_to_0]

    if len(conf_0_to_1_masked) > 0:
        _, topk_local_0_to_1 = torch.topk(
            conf_0_to_1_masked,
            k=min(top_k, len(conf_0_to_1_masked)),
            largest=True
        )
        paths_0_to_1 = allpath[mask_0_to_1][topk_local_0_to_1].tolist()
    else:
        paths_0_to_1 = []

    if len(conf_1_to_0_masked) > 0:
        _, topk_local_1_to_0 = torch.topk(
            conf_1_to_0_masked,
            k=min(top_k, len(conf_1_to_0_masked)),
            largest=True
        )
        paths_1_to_0 = allpath[mask_1_to_0][topk_local_1_to_0].tolist()
    else:
        paths_1_to_0 = []

    if class_names and len(paths_0_to_1) > 0:
        print(f"\nTop {len(paths_0_to_1)} suspicious 0->1 errors (model confident in '{class_names[1]}'):")
        for i, p in enumerate(paths_0_to_1[:3]):
            idx = np.where((all_target == 0) & (all_pred == 1) & (allpath == p))[0][0]
            conf_val = conf_0_to_1_full[idx].item()
            print(f"  [{i+1}] {p} | conf={conf_val:.4f}")

    def get_unique_two_levels(paths: list) -> list:
        two_levels = set()
        for p in paths:
            parts = Path(p).parts
            if len(parts) >= 2:
                two_levels.add("/".join(parts[:2]))
            elif len(parts) == 1:
                two_levels.add(parts[0])
        return sorted(two_levels)

    return get_unique_two_levels(paths_0_to_1), get_unique_two_levels(paths_1_to_0)



def build_scheduler(args, optimizer, n_iter_per_epoch):
    """Build the scheduler
    The CosineLRScheduler schedules every iter in every epoch, so it needs n_iter_per_epoch instead of epochs

    Args:
        optimizer
        n_iter_per_epoch:
    """
    num_steps = int(args.epochs * n_iter_per_epoch)
    warmup_steps = int(args.warmup_epochs * n_iter_per_epoch)

    # use the Cosine LR Scheduler for training
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=args.min_lr,
        warmup_lr_init=args.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    return lr_scheduler


class DMIN(nn.Module):
    """The Dynamic Multi Instance Normalization Module
    This module dynamicly normlizes the instances and bags

    Args:
        optimizer
        n_iter_per_epoch:
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(DMIN, self).__init__()

        # init the DMIN module
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))

        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):

        x = rearrange(x, 'b c n -> b n c')
        # calculate the mean and var of input
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        # caculated the weighted mean and var according to the learned weights
        mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
        var = var_weight[0] * var_in + var_weight[1] * var_ln

        # regulize x according to the calculated results
        x = (x-mean) / (var+self.eps).sqrt()
        x = x * self.weight + self.bias
        x = rearrange(x, 'b n c -> b c n')

        return x
