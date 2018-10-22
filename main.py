# coding=utf-8
import numpy
from matplotlib import pyplot as pt
import math
import random

# 产生数据块1
def GeneratePointInCycle2(point_num, radius):
    X =[]
    Y= []
    for i in range(1, point_num+1):
        theta = random.random()*2* numpy.pi;
        r = random.uniform(radius,0.5)
        x = math.sin(theta)* (r**0.5)
        y = math.cos(theta)* (r**0.5)
        X.append(x)
        Y.append(y)
    return X, Y

# 产生数据块2
def GeneratePointInCycle1(point_num, radius):
    X =[]
    Y= []
    for i in range(1, point_num+1):
        theta = random.random()*2* numpy.pi;
        r = random.uniform(0, radius)
        x = math.sin(theta)* (r**0.5)
        y = math.cos(theta)* (r**0.5)
        X.append(x)
        Y.append(y)
    return X, Y

# 读取文本，进行分割，返回分割的矩阵X、Y；Y为标签矩阵
def read_data(txt):
    data = open(txt).readlines()
    length = len(data)
    list_tolist = []
    for line in data:
        line = line.strip().split(' ')
        list_tolist.append(line)
    # 把list变成矩阵,list_array是一个完整的数据阵，包含x,y
    list_array = numpy.array(list_tolist)
    x = numpy.ones((length, 3))
    x[:,0] = list_array[:,0]
    x[:,1] = list_array[:,1]
    # print(xdata) # X数据准备好
    y = numpy.array(list_array[:,2])
    return x, y

# 批量梯度下降法
def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrains = x.transpose()                             #得到它的转置
    for i in range(0, maxIterations):
        hypothesis = sigmoid(numpy.dot(x, theta)) #修改
        loss = hypothesis - y
        # cost = 1/2*numpy.dot(loss, loss)
        # cost = (loss * loss).sum()
        cost = costFunction(theta,x,y)
        print('cost: ',cost)
        gradient = numpy.dot(xTrains, loss) / m             #对所有的样本进行求和，然后除以样本数
        theta = theta - alpha * gradient
    return theta

# 展示分割结果
def show(X, Y,weights):
    n = len(Y)
    x1 = []  # 下标刚好对应该图像的标签
    y1 = []
    x0 = []
    y0 = []
    for i in range(n):
        if int(Y[i]) == 1:
            x1.append(X[i, 0])
            y1.append(X[i, 1])
        else:
            x0.append(X[i, 0])
            y0.append(X[i, 1])
    x_line = numpy.arange(-0.5, 1.2, 0.1)
    y_line = (-weights[2] - weights[0] * x_line) / weights[1]
    fig = pt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=30, c='r', marker='o', label='Admitted')  # 上半部分
    ax.scatter(x0, y0, s=30, c='g', marker='x', label='Admitted')  # 下半部分
    pt.plot(x_line, y_line)
    pt.xlabel('x', size=10)
    pt.ylabel('y', size=10)
    pt.show()

def sigmoid(z):
    return 1.0 / (1 + numpy.exp(-z))


def costFunction(weight, x, y):
    first = numpy.dot(-y, numpy.log(sigmoid(numpy.dot(x, weight))))
    second = numpy.dot((1 - y), numpy.log(1 - sigmoid(numpy.dot(x, weight))))
    return numpy.sum((first - second) / len(x))

# 使用测试集，测试分割的准确性
def test(theta):
    txt = 'test.txt'
    x,y = read_data(txt)
    result = numpy.dot(x, theta)
    result = sigmoid(result)
    #对每一列进行测试
    count = 0
    for i in range(len(x)):
        print(y[i],result[i])
        if(result[i] >0.5 and y[i] == '1')or (result[i] <0.5 and y[i]=='0'):
            count = count+1  #统计准确率
            print(count)
    return float(count/len(x))

txt = 'data.txt'
x, y = read_data(txt) # 从文件中读取正确的x,y
y = y.astype('float64')
m, n = numpy.shape(x)
theta = numpy.ones(n)

alpha = 0.3 # 步长0.05 -0.14
maxIteration = 200000 #迭代的次数

'''
# 无惩罚项
theta = batchGradientDescent(x, y, theta, alpha, m, maxIteration)
print(theta)
accuracy = test(theta) # 测试测试集的准确率
print('测试集的准确率是：',accuracy)
show(x,y,theta)
'''
# 有惩罚项
langbuda = numpy.e ** -20
x_langbuda = x+langbuda * (numpy.ones((m, n)))
theta = batchGradientDescent(x_langbuda, y, theta, alpha, m, maxIteration)
print(theta)
accuracy = test(theta) # 测试测试集的准确率
print('测试集的准确率是：',accuracy)
show(x,y,theta)

