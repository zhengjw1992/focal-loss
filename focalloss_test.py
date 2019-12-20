from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import os,sys,random,time
import argparse

from focalloss import *


floss = FocalLoss(gamma=0)
crossloss = nn.CrossEntropyLoss()
nllloss = nn.NLLLoss2d()

start_time = time.time()
maxe = 0
for i in range(1000):
    x = torch.rand(12800,2)*random.randint(1,10)
    # x = Variable(x.cuda())
    x = Variable(x)
    l = torch.rand(12800).ge(0.1).long()
    # l = Variable(l.cuda())
    l = Variable(l)
    print('x的维度是：',x.shape)
    print('l的维度是：',l.shape)

    # output0 = FocalLoss(gamma=0)(x,l)
    # output1 = nn.CrossEntropyLoss()(x,l)
    # a = output0.data[0]
    # b = output1.data[0]
    output0 = floss(x,l)
    output1 = crossloss(x,l)
    a = output0.data
    b = output1.data
    if abs(a-b)>maxe:
        maxe = abs(a-b)
print('time:',time.time()-start_time,'max_error:',maxe)


start_time = time.time()
maxe = 0
for i in range(100):
    x = torch.rand(128,1000,8,4)*random.randint(1,10)
    # x = Variable(x.cuda())
    x = Variable(x)
    l = torch.rand(128,8,4)*1000    # 1000 is classes_num
    l = l.long()
    # l = Variable(l.cuda())
    l = Variable(l)
    print('x的维度是：',x.shape)
    print('l的维度是：',l.shape)

    # output0 = FocalLoss(gamma=0)(x,l)
    # output1 = nn.NLLLoss2d()(F.log_softmax(x),l)
    # a = output0.data[0]
    # b = output1.data[0]
    output0 = floss(x,l)
    output1 = nllloss(F.log_softmax(x),l)
    a = output0.data
    b = output1.data
    if abs(a-b)>maxe:
        maxe = abs(a-b)
print('time:',time.time()-start_time,'max_error:',maxe)
