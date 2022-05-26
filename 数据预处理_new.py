# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 20:57:26 2022

@author: 30601
"""

# 使用 read_parquet 加载parquet文件
import pandas as pd
import numpy as np
import torch
from pandas import read_parquet
stock_row = read_parquet("equity_dataset_all_exst.parq.gz")
stock_row.describe()#检查数据的分布，查看高斯和非高斯
head = stock_row.head()
stock_row.info()
a = stock_row.groupby('wind_code').count()
b = stock_row.groupby('trading_date').count()
b.iloc[-1125,:]
#2016-08-15

stock_all = stock_row.drop(columns=['amount'])
dic = {'trading_date':'time','wind_code':'code','open_price':'open','close_price':'close',
       'high_price':'high','low_price':'low','pct_change':'return1'}
stock_all.rename(columns = dic,inplace = True)
stock_all['time'] = pd.to_datetime(stock_all['time'])
#b.iloc[-1125,:]-->2016-08-15
stock_all = stock_all[stock_all['time']>= '2016-08-15']
c = stock_all.groupby('code').count()
d = stock_all.groupby('time').count()
#选出时间跨度合适的股票
c['sign'] = c['time'].apply(lambda x: 0 if x==1125 else 1)
stock_name = c[c['sign']==0].reset_index()['code']

#计算y值
equity = stock_all[stock_all['code']==stock_name[0]]
equity['post_close'] = equity['close']*equity['adj_factor']
equity['return10'] = (equity['post_close'].shift(-10)/equity['post_close'] -1)*100
equity['return5'] = (equity['post_close'].shift(-5)/equity['post_close'] -1)*100
equity['return3'] = (equity['post_close'].shift(-3)/equity['post_close'] -1)*100
for name in stock_name[1:]:
    one_equity1 = stock_all[stock_all['code']==name]
    one_equity1['post_close'] = one_equity1['close']*one_equity1['adj_factor']
    one_equity1['return10'] = (one_equity1['post_close'].shift(-10)/one_equity1['post_close'] -1)*100
    one_equity1['return5'] = (one_equity1['post_close'].shift(-5)/one_equity1['post_close'] -1)*100
    one_equity1['return3'] = (one_equity1['post_close'].shift(-3)/one_equity1['post_close'] -1)*100
    equity = pd.concat((equity,one_equity1),axis=0)
    

#截面标准化
minmax = equity.loc[:,['time','open','high','low','close','volume','return1','vwap','turn','free_turn']]
minmax['time'] = minmax['time'].dt.strftime("%Y-%m-%d")
minmax = minmax.groupby('time').apply(lambda x : (x-np.min(x))/(np.max(x)-np.min(x))) 
#如果先前分布是高斯的可以，若不是高斯可以尝试正态标准化，不同特征考虑不同的标准化
#个股的收益率分布情况，label可以不用作做标准化
equity_x = equity[['time','code','open','high','low','close','volume','return1','vwap','turn','free_turn']]
equity_x_minmax = pd.concat((equity_x.iloc[:,:2],minmax),axis = 1)
equity_y = equity[['time','code','return10','return5','return3']]
# equity_x.to_csv('C://Users//30601//Desktop//equity_dataset_all_exst.parq//equity_x.csv')
# equity_x_minmax.to_csv('C://Users//30601//Desktop//equity_dataset_all_exst.parq//equity_x_minmax.csv')
# equity_y.to_csv('C://Users//30601//Desktop//equity_dataset_all_exst.parq//equity_y.csv')
equity_x = pd.read_csv('C://Users//30601//Desktop//equity_dataset_all_exst.parq//equity_x.csv',index_col=0)
equity_x_minmax = pd.read_csv('C://Users//30601//Desktop//equity_dataset_all_exst.parq//equity_x_minmax.csv',index_col=0)
equity_x.iloc[:,2:] = equity_x.iloc[:,2:].astype('float32')
equity_x_minmax.iloc[:,2:] = equity_x_minmax.iloc[:,2:].astype('float32')
equity_y = pd.read_csv('C://Users//30601//Desktop//equity_dataset_all_exst.parq//equity_y.csv',index_col=0)
equity_y.iloc[:,2:] = equity_y.iloc[:,2:].astype('float32')
stock_name = equity_x['code'].drop_duplicates()
def get_picture(stock_x):
    #输入是个股量价数据，12-22年，第一行为12年
    stock = stock_x.T
    X = torch.zeros(size=(1,1,9,30))
    for i in range(0,stock.shape[1]-29):
        a = torch.from_numpy(stock.iloc[:,i:i+30].values)
        a = a.reshape(1,1,9,30)
        X = torch.vstack((X,a))
    return X[1:,:,:,:]

def get_whole_stock(stock_x = equity_x, stock_name = stock_name,stock_time=1125):
    temp_x = torch.zeros(size=(stock_time-29-10,1,9,30))
    for name in stock_name:
        stock = stock_x[stock_x['code'] == name] #nan值填补？
        stock.set_index('time',inplace = True)
        stock = stock.iloc[:,1:].sort_index()
        pictures_x = get_picture(stock)
        temp_x = torch.cat((temp_x,pictures_x[:-10,:,:,:]),dim=1)
    return temp_x[:,1:,:,:]

x = get_whole_stock(stock_x = equity_x, stock_name = stock_name, stock_time=1125)
x_minmax = get_whole_stock(stock_x = equity_x_minmax, stock_name = stock_name,stock_time=1125)

def get_whole_stocky(stock_y = equity_y,stock_name=stock_name,stock_time=1125):
    temp_y = torch.zeros(size=(stock_time-29-10,1,3))
    for name in stock_name:
        stock = stock_y[stock_y['code'] == name] #nan值填补？
        stock.set_index('time',inplace = True)
        stock = stock.iloc[:,1:].sort_index()
        stock_y = stock[29:][:-10]#数据前29天没有对应的y,后10天nan值
        pictures_y = torch.from_numpy(stock_y.values).reshape(stock_y.shape[0],1,3)
        temp_y = torch.cat((temp_y,pictures_y),dim=1)
    return temp_y[:,1:,:]

y = get_whole_stocky(stock_y = equity_y, stock_name=stock_name, stock_time=1125)

temp_y = torch.zeros(size=(1125-29-10,1,3))
for name in stock_name:
    stock = equity_y[equity_y['code'] == name] #nan值填补？
    stock.set_index('time',inplace = True)
    stock = stock.iloc[:,1:].sort_index()
    stock_y = stock[29:][:-10]#数据前29天没有对应的y,后10天nan值
    pictures_y = torch.from_numpy(stock_y.values).reshape(stock_y.shape[0],1,3)
    temp_y = torch.cat((temp_y,pictures_y),dim=1)
y = temp_y[:,1:,:]

torch.save(x , r"C:\Users\30601\Desktop\pictures_x.pt")
torch.save(x_minmax , r"C:\Users\30601\Desktop\pictures_x_minmax.pt")
torch.save(y , r"C:\Users\30601\Desktop\delta10_5_3_y.pt")
