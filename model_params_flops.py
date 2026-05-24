from models import *
from thop import profile
import torch
import copy

def model_params_flops(model):
    model = copy.deepcopy(model)
    print('==========================================================================')
    input = torch.randn(1, 3, 84, 84)
    macs, params = profile(model, inputs=(input,))
    print('==========================================================================')
    print('Total params:: {:.3f} M\n'
          'Total FLOPs: {:.3f}  GFLOPs'.format(params/10**6, macs/10**9))
    print('==========================================================================')

if __name__=='__main__':
    model = resnet18(num_classes=200)
    model_params_flops(model)

