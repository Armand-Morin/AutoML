from collections import OrderedDict

import torch
import torch.nn as nn

import ops
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice

channels= 4
stride=2
choice_keys=['mutation1','mutation2','mutation3','mutation4','mutation5']
opsl = nn.ModuleList()

for i in range(5):
    opsl.append(
        LayerChoice(OrderedDict([
            ("maxpool", ops.PoolBN('max', channels, 3, stride, 1, affine=False)),
            ("avgpool", ops.PoolBN('avg', channels, 3, stride, 1, affine=False)),
            ("skipconnect", nn.Identity() if stride == 1 else ops.FactorizedReduce(channels, channels, affine=False)),
            ("sepconv3x3", ops.SepConv(channels, channels, 3, stride, 1, affine=False)),
            ("sepconv5x5", ops.SepConv(channels, channels, 5, stride, 2, affine=False)),
            ("dilconv3x3", ops.DilConv(channels, channels, 3, stride, 2, 2, affine=False)),
            ("dilconv5x5", ops.DilConv(channels, channels, 5, stride, 4, 2, affine=False))
        ]), label=choice_keys[i]))
drop_path = ops.DropPath()

print(opsl)