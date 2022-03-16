import os

import torch
import torch.nn as nn
from torchvision import models

# 一个pytorch模型只要包含两个部分：模型结构和权重。其中模型是继承nn.Module的类， 权重的数据结构是一个字典（key是层名，value是权重向量).存储也由此分为两种形式：存储整个模型（包括结构和权重），和只存储模型权重.
model = models.resnet152(pretrained=True)

# 保存整个模型
# torch.save(model, save_dir)
# torch.save(model.state_dict, save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 如果是多卡改成类似0，1，2
model = model.cuda() # 单卡
model = torch.nn.DataParallel(model).cuda() # 多卡

# 保存+读取整个模型
torch.save(model, save_dir)
load_model = torch.load(save_dir)
load_model.cuda()

# 保存+读取模型权重
torch.save(model.state_dict(), save_dir)
load_dict = torch.load(save_dir)
loaded_model = models.resnet152() # 注意这里需要对模型结构有定义
loaded_model.state_dict = load_dict
loaded_model.cuda()


# 对于加载模型，有以下几种思路:
# 去除字典里的module麻烦，往model里添加module简单

import os
import torch
from torchvision import models

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

load_dict = torch.load(save_dir)

