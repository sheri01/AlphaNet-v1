# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:30:01 2022

@author: 30601
"""

import torch
from torch import nn
import numpy as np
from scipy.stats import pearsonr


def cor(feature, d, col):
    length = feature.shape[0]
    temp = np.zeros([int(length*(length-1)/2), 1])
    index = 0
    for row1 in range(0, length):
        for row2 in range(row1+1, length):
            temp[index, 0] = pearsonr(feature[row1, col:col+d], feature[row2, col:col+d])[1]
            index += 1
    return temp


# 特征维度求协方差
# col为起始列数
def cov(feature, d, col):
    length = feature.shape[0]
    temp = np.zeros([int(length*(length-1)/2),1])
    index = 0
    for row1 in range(0, length):
        for row2 in range(row1+1, length):
            temp[index, 0] = np.cov(feature[row1, col:col+d], feature[row2, col:col+d])[0, 1]
            index += 1
    return temp

# 过去 d 天 X 值构成的时序数列和 Y 值构成的时序数列的相关系数
def ts_corr(x, d, stride=10):
    def t_corr(stock):
        stock = stock.cpu().numpy()
        length = stock.shape[0]
        width = stock.shape[1]
        feature = np.zeros([int(length * (length - 1) / 2), int((width - d) / stride + 1)])
        # 时间维度求相关系数
        index = 0
        for col in range(0, width - d + 1, stride):
            feature[:, index] = cor(stock, d, col).ravel()
            index += 1
        feature = torch.from_numpy(feature).to(torch.float32).cuda()
        return feature
    corr = torch.zeros(size = (1,1,36,3)).cuda()#经过自定义网络层的输出大小，可修改
    for i in range(0,x.shape[0]):
        x1 = t_corr(x[i,0,:,:])#将样本拆分
        x1 = x1.reshape(1,1,x1.shape[0],x1.shape[1])#reshape成可拼接的形状
        corr = torch.vstack((corr,x1))
    return corr[1:,:,:,:]

# 过去 d 天 X 值构成的时序数列和 Y 值构成的时序数列的协方差
def ts_cov(x, d, stride=10):
    def t_cov(stock):
        stock = stock.cpu().numpy()
        stock = np.array(stock)
        length = stock.shape[0]
        width = stock.shape[1]
        covxy = np.zeros([int(length * (length - 1) / 2), int((width - d) / stride + 1)])
        # 时间维度求协方差
        index = 0
        for col in range(0, width - d + 1, stride):
            covxy[:, index] = cor(stock, d, col).ravel()
            index += 1
        covxy = torch.from_numpy(covxy).to(torch.float32).cuda()
        return covxy
    cov = torch.zeros(size = (1,1,36,3)).cuda()#经过自定义网络层的输出大小，可修改
    for i in range(0,x.shape[0]):
        x1 = t_cov(x[i,0,:,:])#将样本拆分
        x1 = x1.reshape(1,1,x1.shape[0],x1.shape[1])#reshape成可拼接的形状
        cov = torch.vstack((cov,x1))
    return cov[1:,:,:,:]

# 过去 d 天 X 值构成的时序数列的标准差
def ts_stddev(x, d, stride=10):
    def t_stddev(stock):
        stock = stock.cpu().numpy()
        length = stock.shape[0]
        width = stock.shape[1]
        stdx = np.zeros([length, int((width - d) / stride + 1)])
        # col为起始列数
        index = 0
        for col in range(0, width - d + 1, stride):
            for row in range(0, length):
                stdx[row, index] = np.std(stock[row, col:col+d])
            index += 1
        stdx = torch.from_numpy(stdx).to(torch.float32).cuda()
        return stdx
    stddev = torch.zeros(size = (1,1,9,3)).cuda()#经过自定义网络层的输出大小，可修改
    for i in range(0,x.shape[0]):
        x1 = t_stddev(x[i,0,:,:])#将样本拆分
        x1 = x1.reshape(1,1,x1.shape[0],x1.shape[1])#reshape成可拼接的形状
        stddev = torch.vstack((stddev,x1))
    return stddev[1:,:,:,:]

# 过去 d 天 X 值构成的时序数列的平均值除以标准差
def ts_zscore(x, d, stride=10):
    def t_zscore(stock):
        stock = stock.cpu().numpy()
        stock = np.array(stock)
        length = stock.shape[0]
        width = stock.shape[1]
        z_score = np.zeros([length, int((width - d) / stride + 1)])
        # col为起始列数
        index = 0
        for col in range(0, width - d + 1, stride):
            for row in range(0, length):
                mean = np.mean(stock[row, col:col+d])
                stdx = np.std(stock[row, col:col+d])
                z_score[row, index] = mean/stdx
            index += 1
        z_score = torch.from_numpy(z_score).to(torch.float32).cuda()
        return z_score
    zs = torch.zeros(size = (1,1,9,3)).cuda()#经过自定义网络层的输出大小，可修改
    for i in range(0,x.shape[0]):
        x1 = t_zscore(x[i,0,:,:])#将样本拆分
        x1 = x1.reshape(1,1,x1.shape[0],x1.shape[1])#reshape成可拼接的形状
        zs = torch.vstack((zs,x1))
    return zs[1:,:,:,:]


# d 天以前的 X 值
def delay(feature, d):
    return feature[-(d+1)]

# (X - delay(X, d))/delay(X, d)-1
def ts_return(x, d, stride=10):
    def t_return(stock):
        stock = stock.cpu().numpy()
        stock = np.array(stock)
        length = stock.shape[0]
        width = stock.shape[1]
        res = np.zeros([length, int((width - d) / stride + 1)])
        # col为起始列数
        index = 0
        for col in range(0, width - d + 1, stride):
            for row in range(0, length):
                res[row, index] = np.sum(stock[row, col:col+d])
            index += 1
        res = torch.from_numpy(res).to(torch.float32).cuda()
        return res
    re = torch.zeros(size = (1,1,9,3)).cuda()#经过自定义网络层的输出大小，可修改
    for i in range(0,x.shape[0]):
        x1 = t_return(x[i,0,:,:])#将样本拆分
        x1 = x1.reshape(1,1,x1.shape[0],x1.shape[1])#reshape成可拼接的形状
        re = torch.vstack((re,x1))
    return re[1:,:,:,:]

# 过去 d 天 X 值构成的时序数列的加权平均值
# 权数为 d, d – 1, …, 1(权数之和应为 1，需进行归一化处理)，其中离现在越近的日子权数越大
def ts_decaylinear(x, d, stride=10):      
    def t_decaylinear(stock):
        stock = stock.cpu().numpy()
        stock = np.array(stock)
        length = stock.shape[0]
        width = stock.shape[1]
        weight = [w for w in range(d, 0, -1)]
        weight = weight/np.sum(weight)
        dl = np.zeros([length, int((width - d) / stride + 1)])
        # col为起始列数
        index = 0
        for col in range(0, width - d + 1, stride):
            for row in range(0, length):
                dl[row, index] = np.dot(stock[row, col:col + d], weight)
            index += 1
        dl = torch.from_numpy(dl).to(torch.float32).cuda()
        return dl
    de = torch.zeros(size = (1,1,9,3)).cuda()#经过自定义网络层的输出大小，可修改
    for i in range(0,x.shape[0]):
        x1 = t_decaylinear(x[i,0,:,:])#将样本拆分
        x1 = x1.reshape(1,1,x1.shape[0],x1.shape[1])#reshape成可拼接的形状
        de = torch.vstack((de,x1))
    return de[1:,:,:,:]

def cat2(x,y):
    #x第三维度比y长，将y补充为与x一样大小，补充元素用0填
    a = torch.zeros(size=(y.shape[0],y.shape[1],x.shape[2]-y.shape[2],y.shape[3])).cuda()
    a = torch.cat((y,a),dim = 2)
    return a

class feature_catch(nn.Module):#特征提取层
    def __init__(self,  d, stride):
        super().__init__()
        self.d = d
        self.stride = stride
    def forward(self, x):
        conv1 = ts_corr(x, self.d , self.stride)
        conv2 = ts_cov(x, self.d , self.stride)
        conv3 = ts_stddev(x, self.d , self.stride)
        conv3 = cat2(conv1,conv3)#将conv3变成与conv1一样的形状
        conv4 = ts_zscore(x, self.d , self.stride)
        conv4 = cat2(conv1,conv4)
        conv5 = ts_return(x, self.d , self.stride)
        conv5 = cat2(conv1,conv5)
        conv6 = ts_decaylinear(x, self.d , self.stride)
        conv6 = cat2(conv1,conv6)
        return torch.hstack((conv1,conv2,conv3,conv4,conv5,conv6)) #6个输出通道
#这样用零填充conv3-6 是否合理？有什么其他解决办法？

class Pool2d(nn.Module):#三个池化组件
    def __init__(self,  d, stride):
        super().__init__()
        self.d = d
        self.stride = stride
    def forward(self, x):
        t_mean = nn.AvgPool2d((1,self.d), stride=(1,self.stride))
        t_max = nn.MaxPool2d((1,self.d), stride=(1,self.stride))
        ts_mean = t_mean(x)
        ts_max = t_max(x)
        ts_min = - t_max(- x)
        return torch.stack((ts_mean,ts_max,ts_min),dim=1) #3个输出通道 

class InceptionA(nn.Module):
    #将 特征提取层-->池化层  和 特征提取层 的输出 按通道维数合在一起
    def __init__(self):
        super(InceptionA,self).__init__()

        self.branch1 = nn.Sequential(            
            feature_catch(d=10, stride=10),
            nn.BatchNorm2d(6),
            nn.Flatten()
        )  
        
        self.branch2 = nn.Sequential(            
            feature_catch(d=10, stride=10),
            nn.BatchNorm2d(6),
            Pool2d(d=3, stride=10),
            nn.BatchNorm3d(3),
            nn.Flatten()
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        return torch.cat((branch1,branch2),dim=1)

#构建神经网络                
branch1 = nn.Sequential(            
    feature_catch(d=10, stride=10),
    nn.BatchNorm2d(6),
    nn.Flatten()
)  

branch2 = nn.Sequential(            
    feature_catch(d=10, stride=10),
    nn.BatchNorm2d(6),
    Pool2d(d=3, stride=10),
    nn.BatchNorm3d(3),
    nn.Flatten()
)

alphanet = nn.Sequential(
    InceptionA(),
    nn.Flatten(),
    nn.Linear(648*2, 30), 
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(30, 1))

# print(alphanet)
#网络结构可视化：
# import hiddenlayer as h
# vis_graph = h.build_graph(alphanet, torch.zeros([bs ,1, 9, 38]))   # 获取绘制图像的对象
# vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
# vis_graph.save("C://Users//30601//Desktop//demo1.png")   # 保存图像的路径

#打印数据在 alphanet 和 branch1、2 每一层输出的形状
# X = torch.rand(size=(64, 1, 9, 30), dtype=torch.float32)
# for layer in alphanet:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape: \t',X.shape)

# Y = torch.rand(size=(64, 1, 9, 30), dtype=torch.float32)
# for layer in branch1:
#     Y = layer(Y)
#     print(layer.__class__.__name__,'output shape: \t',Y.shape)
    
# Z = torch.rand(size=(64, 1, 9, 30), dtype=torch.float32)
# for layer in branch2:
#     Z = layer(Z)
#     print(layer.__class__.__name__,'output shape: \t',Z.shape)   

    
    
    
