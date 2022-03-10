# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 22:30:49 2022

@author: 30601
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from d2l import torch as d2l 
from alphanet_v1_on_gpu import alphanet


def tensor_shuffle(set1, set2):
    """
    输入X和y，返回对应被打乱的结果
    """
# 打乱索引从而打乱数据集
    length = set1.shape[0]
    index = [i for i in range(length)]
    np.random.shuffle(index)
    set1 = set1[index]
    set2 = set2[index]
    return set1, set2

#导入数据
# 2207: 2083,half of year:125
X_all = torch.load(r"002207.XSHE_pictures_x.pt").type(torch.FloatTensor)
Y_all = torch.load(r"002207.XSHE_pictures_y.pt").type(torch.FloatTensor)

X = X_all[:1500,:,:,:]
Y = Y_all[:1500,:]
x, y = tensor_shuffle(X, Y)

#8:2
train_x = x[:1200,:,:,:]
train_y = y[:1200,:]
test_data_x = x[1200:,:,:,:]
test_data_y = y[1200:,:]

train_data = TensorDataset(train_x, train_y)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

#start to train
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.trunc_normal_(m.weight)
#cuda：0
device = d2l.try_gpu() 

def train_net(alphanet,train_loader,test_data_x , test_data_y,lr = 0.001,times = 1):
    # RMSProp，0.0001
    alphanet.to(device)
    optimizer = torch.optim.RMSprop(alphanet.parameters(),lr = lr)
    loss_func = torch.nn.MSELoss()
    test_data_x , test_data_y = test_data_x.to(device), test_data_y.to(device)    
    print('training on', device)
    test_mse0 = 1 
    for epoch in range(200):
        if epoch > 120: #减小学习率
            optimizer = torch.optim.RMSprop(alphanet.parameters(),lr = 0.0001)  
         # print("epoch:", epoch)
         # 每一轮都遍历一遍数据加载器 
        for step, (x, y) in enumerate(train_loader):
             # 前向计算->计算损失函数->(从损失函数)反向传播->更新网络
             optimizer.zero_grad()   
             x, y = x.to(device), y.to(device)
             predict = alphanet(x)
             loss = loss_func(predict,y)
             loss.backward()     
             optimizer.step()   
   
        if epoch< 3 or epoch > 98:
            test_predict = alphanet(test_data_x)
            test_mse = loss_func(test_predict,test_data_y)            
         # 控制台输出一下
            print("epoch:{}, train_mse:{:.2},test_mse:{:.2}".format(epoch + 1, loss.item(), test_mse.item()))
            if test_mse < test_mse0: #验证集误差小于上一轮epoch，则保存模型
                test_mse0 = test_mse
                print('save model')
                torch.save(alphanet.state_dict(), 
                           "model_param{}.pth".format(times)) 
                        
for i in range(0,10):
    print('第{}次训练'.format(i+1))
    alphanetv1 = alphanet
    alphanetv1.apply(init_weights) #初始化参数
    train_net(alphanetv1,train_loader,test_data_x , test_data_y,lr = 0.001,times = i + 1)                                                                

