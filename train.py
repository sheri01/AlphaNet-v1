# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 22:30:49 2022

@author: 30601
"""
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from d2l import torch as d2l 
from alphanet_v1 import alphanet
from torch import nn
# from pytorchtools import EarlyStopping
#导入数据
# 股票代码：002207.XSHE；12-22年总样本数 2083
X_all = torch.load(r"002207.XSHE_pictures_x.pt").type(torch.FloatTensor)
Y_all = torch.load(r"002207.XSHE_pictures_y.pt").type(torch.FloatTensor)

X = X_all[:1500,:,:,:]
Y = Y_all[:1500,].reshape(1500,1)
#6：4 训练：验证
train_x = X[:950,:,:,:]
train_y = Y[:950,:]
test_data_x = X[950:,:,:,:]
test_data_y = Y[950:,:]

train_data = TensorDataset(train_x, train_y)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

#start to train

#权重初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.trunc_normal_(m.weight)
#cuda：0
device = d2l.try_gpu()

def train_net(alphanet,train_loader,test_data_x , test_data_y,lr = 0.001,times = 10):
    alphanet.to(device)
    optimizer = torch.optim.RMSprop(alphanet.parameters(),lr = lr)
    loss_func = torch.nn.MSELoss()
    test_data_x , test_data_y = test_data_x.to(device), test_data_y.to(device)    
    print('training on', device)
    
    #训练和验证集损失小于0.1和1，可以开始保存模型参数了。设置大一点，防止epoch200次中验证集的损失一直很大导致没有保存模型
    loss0 = 0.1
    test_mse0 = 1
    
    for epoch in range(200):
        if epoch > 120: #降低学习率
            optimizer = torch.optim.RMSprop(alphanet.parameters(),lr = 0.0001)  
         # 每一轮都遍历一遍数据加载器 
        for step, (x, y) in enumerate(train_loader):
             # 前向计算->计算损失函数->(从损失函数)反向传播->更新网络
             optimizer.zero_grad()   # 清空梯度
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
            if loss < loss0 and test_mse < test_mse0: #留下200次迭代中，训练集和验证集的损失最小的那个模型参数
                loss0 = loss
                test_mse0 = test_mse
                print('save model')
                torch.save(alphanet.state_dict(), 
                           "model_param{}.pth".format(times)) 
#防止初始参数值对模型训练的影响，重复10次
for i in range(0,10):
    print('第{}次训练'.format(i+1))
    alphanetv1 = alphanet
    alphanetv1.apply(init_weights) #初始化参数
    train_net(alphanetv1,train_loader,test_data_x , test_data_y,lr = 0.001,times = i + 1)                                                               
    
    
    
