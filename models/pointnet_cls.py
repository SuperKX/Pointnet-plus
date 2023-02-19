import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

'''
get_model设计的关键点
1） 继承nn.Module
2） 定义初始化条件、前向传播条件。
'''
class get_model(nn.Module):   # 整个网络接哦古
    # 初始化
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)     # 全连接层nn.Linear的输入输出一般2维，形状通常为[batch_size, size]。如（24，1024）->（24，512）
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    # 前向传播
    def forward(self, x):  # input (24,3,1024)
        # 升维（编码）过程：（n*3）->（1*1024）全连接
        x, trans, trans_feat = self.feat(x)

        # 降维（解码）过程： 1024-512-256-k
        x = F.relu(self.bn1(self.fc1(x)))  # 全连接层-归一化-激活函数
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
