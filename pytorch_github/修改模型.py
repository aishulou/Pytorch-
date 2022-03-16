# 修改模型层

import torch
import torch.nn as nn
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = models.resnet50()
print(net)

# 这里模型结构是为了适配ImageNet预训练的权重，因此最后全连接层（fc)的输出节点数是1000
# 假设我们要用这个resnet模型去做一个10分类的问题，就应该修改模型的fc层， 将其输出节点数替换为10.另外， 我们觉得一层全连接层可能太少了， 想再加一层，可以做如下修改：
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 128)),
                                        ('relu1', nn.ReLU()),
                                        ('dropout1', nn.Dropout(0.5)),
                                        ('fc2', nn.Linear(128, 10)),
                                        ('output', nn.Softmax(dim=1))]))
net.fc = classifier  # 相当于将模型(net)最后名称为“fc”的层替换成名称为“classifier"的结构

# 添加外部输入
# 基本思路是：将原模型添加输入位置前的部分作为一个整体，同时在forward中定义好原模型不变的部分、添加的输入和后续层之间的连接关系，从而完成模型的修改

class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_add = nn.Linear(1001, 10, bias=True)
        self.output = nn.Softmax(dim=1)

    def forward(self, x, add_variable):
        x = self.net(x)
        x = torch.cat((self.dropout(self.relu(x)), add_variable.unsqueeze(1)), 1)
        x = self.fc_add(x)
        x = self.output(x)
        return x


# 实例化模型
import torchvision.models as models
net = models.resnet50()
models = Model(net).to(device)

