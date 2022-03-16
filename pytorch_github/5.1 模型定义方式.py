import collections

import torch
import torch.nn as nn
from collections import OrderedDict  # 根据放入元素的先后顺序进行排序，存储排好顺序的字典


# 当模型的前向计算为简单串联各个层的计算时， Sequential类可以通过更加简单的方式定义模型。它可以接收一个子模块的有序字典（OrderedDict)或者一系列子模块作为参数来逐一添加Module的实例，
# 而模型的前向计算就是将这些实例按添加的顺序逐一计算。结合Sequential的定义和方式来加以理解
class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):  # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key.module)  # add_module会将module添加进self.modules（一个OrderedDict)
        else:  # 传入的是一些module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        # self._modules返回一个OrderedDict，保证会按照成员添加时的顺序遍历
        for module in self._modules.values():
            input = module(input)
        return input


# Sequential定义模型的方式：只需要将模型的层按顺序排列起来即可
# 直接排列
net1 = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print(net1)

# 使用OrderedDict
net2 = nn.Sequential(collections.OrderedDict([
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 10))
]))
print(net2)

"""
   使用Sequential定义模型的好处在于简单、易读， 同时使用Sequential不需要再写forward。但使用Sequential会时模型定义丧失灵活性
"""

# ModuleList接收一个子模块的（或层，需属于nn.Module类）的列表作为输入， 然后也可以类似List那样进行append和extend操作
# 同时，子模块或层的权重也会自动添加到网络中来
net3 = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net3.append(nn.Linear(256, 10))  # 类似list的append操作
print(net3[-1])
print(net3)

# 注意：nn.ModuleList并没有定义一个网络，它只是将不同的模块存储在一起。ModuleList中元素的先后顺序并不代表其在网络中的真实位置顺序，需要经过forward函数指定各个层的先后顺序才算完成了模型定义
# 用for循环即可
class model(nn.Module):
    def __init__(self, modulelist):
        super(model, self).__init__()
        self.modulelist = nn.ModuleList

    def forward(self, x):
        for layer in self.modulelist:
            x = layer(x)
        return x

#对应模块为nn.ModuleDict().ModuleDict和ModuleList的作用类似，只是ModuleDict能够更方便地为神经网络的层添加名称
net4 = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net4['output'] = nn.Linear(256, 10)  # 添加
print(net4['linear'])  # 访问
print(net4.output)
print(net4)






