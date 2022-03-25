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

#导入模型
import alphanet_v1_torch
feature_catch = alphanet_v1_on_gpu.feature_catch(10, 10)
alphanet = nn.Sequential(
    alphanet_v1_on_gpu.catFeature(),
                    nn.Flatten(),
                    nn.Linear(648*2, 30), 
                    nn.Dropout(0.5),
                    nn.ELU(),
                    nn.Linear(30, 1)
    )
#定义训练函数
def train_net(alphanet,train_loader,valid_data_x , valid_data_y,lr = 0.001,times = 1,k = 0):
    logger = SummaryWriter(log_dir="{}//log{}".format(k, times))
    alphanet.to(device)
    optimizer = torch.optim.RMSprop(alphanet.parameters(),lr = lr)
    loss_func = torch.nn.MSELoss()
    valid_data_x , valid_data_y = valid_data_x.to(device), valid_data_y.to(device)    
    log_step_interval = 19
    print('training on', device)
    valid_mse0 = 1
    save_sign = False #用来判断是否保存了训练好的模型
    n = 0
    for epoch in range(500):
        # print("epoch:", epoch) 
        if epoch > 50:
            optimizer = torch.optim.RMSprop(alphanet.parameters(),lr = 0.0001)
         # 每一轮都遍历一遍数据加载器 
        for step, (x, y) in enumerate(train_loader):
            # 前向计算->计算损失函数->(从损失函数)反向传播->更新网络
            optimizer.zero_grad()   # 清空梯度（可以不写）
            x, y = x.to(device), y.to(device)
            predict = alphanet(x)
            loss = loss_func(predict,y)
            loss.backward()     # 反向传播计算梯度
            optimizer.step()    # 更新网络
        # 在验证集上计算mse
        valid_predict = alphanet(valid_data_x)
        valid_mse = loss_func(valid_predict,valid_data_y) 
        print("epoch:{}, train_mse:{:.2}, valid_mse:{:.2}".format(epoch + 1, loss.item(), valid_mse.item()))
        global_iter_num = epoch * len(train_loader) + step + 1  # 计算当前是从训练开始时的第几步(全局迭代次数)
        if global_iter_num % log_step_interval == 0:
           # 添加的第一条日志：损失函数-全局迭代次数
           logger.add_scalar("train loss", loss.item() ,global_step=global_iter_num)
           # 添加第二条日志：mse-全局迭代次数
           logger.add_scalar("valid_mse", valid_mse.item(), global_step=global_iter_num)
        n += 1 #记下有多少次epoch模型没有改善
        if valid_mse < valid_mse0: #验证集损失改善就保存模型
            save_sign = True
            valid_mse0 = valid_mse
            print('save model')
            torch.save(alphanet.state_dict(), 
                        "{}//model_param_{}.pth".format(k, times)) 
            n = 0 #改善一次n清零
        if n >= 20 and save_sign == True:
            print('early stoping')
            break   
        elif n >= 20 and save_sign == False: #设置早停，20次没有改善就停止
            print('early stoping;save model')
            torch.save(alphanet.state_dict(), 
                       "{}//model_param_{}.pth".format(k, times)) 
            break
    return

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

#开始训练，多只股
X_all = torch.load(r"pictures_x.pt").type(torch.FloatTensor)
Y_all = torch.load(r"pictures_y.pt").type(torch.FloatTensor)

#半年滚动训练，每一个滚动模型训练10次                                                                                
#2440的交易日数量
for k in range(0,8):
    n=125
    #k最大为8，第八次滚动的测试集没有125个交易日
    X = X_all[n*k:1500+n*k-10,:,:,:].flatten(start_dim=0, end_dim=1).unsqueeze(1)
    Y = Y_all[n*k:1500+n*k-10,:].flatten(start_dim=0, end_dim=1).unsqueeze(1)
    X_test = X_all[1500+n*k + 125:1500 + n*(k+1),:,:,:].flatten(start_dim=0, end_dim=1).unsqueeze(1)
    Y_test = Y_all[1500+n*k + 125:1500 + n*(k+1),:].flatten(start_dim=0, end_dim=1).unsqueeze(1)
    x, y = tensor_shuffle(X, Y)
    x, X_test= feature_catch(x),feature_catch(X_test)  #单独提出，加快计算速度
    #8:2 划分
    train_x = x[:1200,:,:,:]
    train_y = y[:1200,:]
    valid_data_x = x[1200:,:,:,:]
    valid_data_y = y[1200:,:]
    train_data = TensorDataset(train_x, train_y)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True,
    )
    for i in range(0,10):
        print('第{}个半年，第{}次训练'.format(k+1, i + 1))
        alphanet.apply(init_weights) #初始化参数
        train_net(alphanet, train_loader, valid_data_x, valid_data_y, lr = 0.001,
                  times = i + 1, k=k+1)     
        X_test , Y_test = X_test.to(device), Y_test.to(device)
        alphanet.load_state_dict(torch.load("{}//model_param_{}.pth".format(k+1, i+1)) )
        test_predict = alphanet(X_test)
        torch.save(test_predict,"{}//test_predict_{}.pt".format(k+1,i+1))

