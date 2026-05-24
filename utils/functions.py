import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.io as sio
import torch
import numpy as np


class stats:
    def __init__(self, path, start_epoch):
        if start_epoch != 0:
           stats_ = sio.loadmat(os.path.join(path,'stats.mat'))
           data = stats_['data']
           content = data[0,0]
           self.trainObj = content['trainObj'][:,:start_epoch].squeeze().tolist()
           self.trainTop1 = content['trainTop1'][:,:start_epoch].squeeze().tolist()
           self.trainTop5 = content['trainTop5'][:,:start_epoch].squeeze().tolist()
           self.valObj = content['valObj'][:,:start_epoch].squeeze().tolist()
           self.valTop1 = content['valTop1'][:,:start_epoch].squeeze().tolist()
           self.valTop5 = content['valTop5'][:,:start_epoch].squeeze().tolist()
           if start_epoch == 1:
               self.trainObj = [self.trainObj]
               self.trainTop1 = [self.trainTop1]
               self.trainTop5 = [self.trainTop5]
               self.valObj = [self.valObj]
               self.valTop1 = [self.valTop1]
               self.valTop5 = [self.valTop5]
        else:
           self.trainObj = []
           self.trainTop1 = []
           self.trainTop5 = []
           self.valObj = []
           self.valTop1 = []
           self.valTop5 = []
    def _update(self, trainObj, top1, top5, valObj, prec1, prec5):
        self.trainObj.append(trainObj)
        self.trainTop1.append(top1.cpu().numpy())
        self.trainTop5.append(top5.cpu().numpy())
        self.valObj.append(valObj)
        self.valTop1.append(prec1.cpu().numpy())
        self.valTop5.append(prec5.cpu().numpy())



def plot_curve(stats, path, iserr=False):
    trainObj = np.array(stats.trainObj)
    valObj = np.array(stats.valObj)
    if iserr:
        trainTop1 = 100 - np.array(stats.trainTop1)
        trainTop5 = 100 - np.array(stats.trainTop5)
        valTop1 = 100 - np.array(stats.valTop1)
        valTop5 = 100 - np.array(stats.valTop5)
        titleName = 'error'
    else:
        trainTop1 = np.array(stats.trainTop1)
        trainTop5 = np.array(stats.trainTop5)
        valTop1 = np.array(stats.valTop1)
        valTop5 = np.array(stats.valTop5)
        titleName = 'accuracy'
    epoch = len(trainObj)
    figure = plt.figure()
    obj = plt.subplot(1,2,1)
    obj.plot(range(1,epoch+1), trainObj,'.-', label = 'train', markersize=2)
    obj.plot(range(1,epoch+1), valObj, '.-',label = 'val', markersize=2)
    plt.xlabel('epoch')
    plt.title('loss')
    handles, labels = obj.get_legend_handles_labels()
    obj.legend(handles[::-1], labels[::-1])
    top1 = plt.subplot(1,2,2)
    top1.plot(range(1,epoch+1),trainTop1,'.-',label = 'train', markersize=2)
    top1.plot(range(1,epoch+1),valTop1,'.-',label = 'val', markersize=2)
    plt.title('top1_'+titleName)
    plt.xlabel('epoch')
    handles, labels = top1.get_legend_handles_labels()
    top1.legend(handles[::-1], labels[::-1])
    # top5 = plt.subplot(1,3,3)
    # top5.plot(range(1,epoch+1),trainTop5,'o-',label = 'train')
    # top5.plot(range(1,epoch+1),valTop5,'o-',label = 'val')
    # plt.title('top5'+titleName)
    # plt.xlabel('epoch')
    # handles, labels = top5.get_legend_handles_labels()
    # top5.legend(handles[::-1], labels[::-1])
    filename = os.path.join(path, 'net-train.pdf')
    figure.savefig(filename, bbox_inches='tight')
    plt.close()

def decode_params(input_params):
    params = input_params[0]
    out_params = []
    _start=0
    _end=0
    for i in range(len(params)):
        if params[i] == ',':
            out_params.append(float(params[_start:_end]))
            _start=_end+1
        _end+=1
    out_params.append(float(params[_start:_end]))
    return out_params

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
