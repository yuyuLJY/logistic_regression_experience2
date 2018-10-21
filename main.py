# coding=utf-8
import numpy
from matplotlib import pyplot as pt
import math

import random
def creat_sindata(number):
    X = numpy.linspace(0.5, 2 * numpy.pi,number, endpoint=True)
    Y = numpy.sin(X)
    mu = 0
    sigma = 0.03
    for i in range(X.size):
        X[i] += random.gauss(mu, sigma)
        Y[i] += random.gauss(mu, sigma)
    return X, Y

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

def read_data():
    data = open('data.txt').readlines()
    length = len(data)
    list_tolist = []
    for line in data:
        line = line.strip().split(' ')
        # print(line)
        list_tolist.append(line)
    # 把list变成矩阵,list_array是一个完整的数据阵，包含x,y
    list_array = numpy.array(list_tolist)
    xdata = numpy.ones((length, 3))
    xdata[:,0] = list_array[:,0]
    xdata[:,1] = list_array[:,1]
    # print(xdata) # X数据准备好
    ydata = numpy.array(list_array[:,2])
    # print(ydata) # y数据已经准备好
    return xdata, ydata

# 批量梯度下降法
def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrains = x.transpose()                             #得到它的转置
    for i in range(0, maxIterations):
        hypothesis = numpy.dot(x, theta)
        loss = hypothesis - y
        # cost = 1/2*numpy.dot(loss, loss)
        # cost = (loss * loss).sum()
        cost = costFunction(theta,x,y)
        print('cost: ',cost)
        gradient = numpy.dot(xTrains, loss) / m             #对所有的样本进行求和，然后除以样本数
        theta = theta - alpha * gradient
    return theta

def show(xArr, yArr,predit_result, weights):
    n = len(yArr)
    dataArr = numpy.array(xArr)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(yArr[i]) == 1:
            xcord1.append(dataArr[i, 0])
            ycord1.append(dataArr[i, 1])
        else:
            xcord2.append(dataArr[i, 0])
            ycord2.append(dataArr[i, 1])
    x = xcord1 + xcord2
    y = [] # 这个是实际情况的y，即图的纵坐标
    # 内圈
    for index in range(len(xcord2)):
        result = (predit_result[index]-xcord2[index]*theta[0]-theta[2])/theta[1]
        y.append(result)
    # 外圈
    for index in range(len(xcord1)):
        result = (predit_result[50+index]-xcord1[index]*theta[0]-theta[2])/theta[1]
        y.append(result)
    fig = pt.figure()
    print(y)
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='r', marker='o', label='Admitted') #外圈
    ax.scatter(xcord2, ycord2, s=30, c='g', marker='x', label='Admitted') #内圈
    ax.scatter(x[0:49], y[0:49], s=30, c='b', marker='o', label='Admitted')
    # ax.plot(x, y)
    pt.xlabel('Score1', size=25)  # 横坐标
    pt.ylabel('Score2', size=25)  # 纵坐标
    pt.title('Logistic', size=30)  # 标题
    pt.show()

def sigmoid(z):
    return 1.0 / (1 + numpy.exp(-z))


def costFunction(weight, x, y):
    first = numpy.dot(-y, numpy.log(sigmoid(numpy.dot(x ,weight))))
    second = numpy.dot((1 - y), numpy.log(1 - sigmoid(numpy.dot(x ,weight))))
    return numpy.sum((first - second) / len(x))

#预测在给定的X,Y下的概率结果 resultY（即等式右边的Y）
def predit(x, theta):
    result = numpy.dot(x,theta)
    result = sigmoid(result)
    return result

x, y = read_data() # 从文件中读取正确的x,y
y = y.astype('float64')
m, n = numpy.shape(x)
theta = numpy.ones(n)


alpha = 0.0001 # 步长0.05 -0.14
maxIteration = 20000 #迭代的次数
theta = batchGradientDescent(x, y, theta, alpha, m, maxIteration)
print(theta)
predit_result = predit(x,theta)
print(predit_result)
show(x,y,predit_result,theta)





'''
R = 0.2
X1 ,Y1 = GeneratePointInCycle1(50, R)
X2 ,Y2 = GeneratePointInCycle2(50, R)
print("打印0标签，内圆")
for i in range(50):
    print(X1[i],Y1[i],0)

print("打印1标签，内外")
for i in range(50):
    print(X2[i],Y2[i],1)

pt.plot(X1, Y1, linestyle='', marker='.')
pt.plot(X2, Y2, linestyle='', marker='.')
pt.show()
'''
'''
X = [-0.3090124789564679, 0.24796310460947704, -0.38591262131913623, -0.39262176241724045, -0.37907225735848543, -0.030765885058076405, 0.21859399281918962, 0.25472396206181414, -0.12094520649811921, 0.20407323691957444, -0.06927009454879593, 0.2924627962947177, -0.29584496926011417, -0.2005686105497144, 0.2610190976828861, 0.20907065357166962, -0.15378701663048963, 0.034111237619486666, 0.11667866688768239, -0.00019469232827664854, -0.4228324497905682, -0.04447699224033848, 0.10065747960823644, 0.27108945986012306, -0.41969532263034187, -0.44048295694268136, 0.04306455752261238, -0.07992675070640877, 0.26926142902778477, -0.24073490022376492, -0.14044460114311336, 0.2767967036796749, -0.1453783632861086, 0.3296191118651354, 0.20029650226775367, -0.23708906381525996, 0.29746785847980983, 0.02591032865087975, -0.12443572177012677, 0.24626986090943576, -0.12846077632658207, 0.20884803792974235, -0.2653528782529987, -0.050065492662742773, 0.32559594684629756, -0.39725329729403924, -0.06308378562464138, -0.1195119190463622, 0.06170470831960199, -0.2816320040744901,0.6013394326614785, -0.5792947964153351, 0.44596466893380354, -0.16037049645009427, -0.47797164634038736, 0.6000352593459954, -0.6486013205213992, 0.1422267766166175, 0.2686401255556255, 0.5642581232235695, 0.4297676411961863, -0.16459933483122943, 0.5099356972406062, 0.5896732547655364, -0.6389711406020706, 0.14060513458619517, 0.47967156561809104, -0.01739403797037134, 0.21173875440970163, 0.20085877421083215, 0.2207007961311297, -0.3836927465631492, 0.4856307604686055, 0.6673860699962517, 0.4925315878860686, 0.5059843012802715, 0.0099188366071535, -0.689368152282499, 0.21360273496507826, 0.35719554866152786, 0.09332556686639423, -0.0042145161510581805, -0.47741982413392164, 0.181251579571579, 0.4487306311172102, 0.40250712947483913, -0.6766932673047411, -0.43314803257674367, -0.15919766693248644, 0.12075315855518542, -0.6069548263867789, 0.07361629629175959, -0.0719902272670657, 0.6446179887990913, 0.27160881344864457, 0.2292886545545242, -0.5370414925952077, 0.2830881449806103, -0.4879528869178687, -0.5005118808097115]
Y = [0.06024684906783993, 0.09530516554930023, -0.14671820184404605, -0.11779350671704222, 0.15199808898023368, -0.3592540506839911, -0.1355207745594358, -0.12537319879467265, 0.04743144078952106, 0.2765700933801971, 0.1549663630339956, -0.1713328912155385, 0.24963174026576046, -0.007827662970292592, 0.16281981510690038, -0.24179615538784283, -0.4095914373139882, -0.3438243793589, -0.18931899733419108, 0.2562866826175828, -0.056527893032724146, -0.4009057426436966, -0.35038045973900905, 0.3067245165589268, 0.09060491338174373, 0.012351734785083167, 0.15034977428397323, -0.26049601569440717, 0.22853775359037196, -0.26443042280699425, 0.02020107599159188, -0.226248241978555, -0.16196331831706676, 0.16324506336230452, -0.1428148289890624, 0.18525420738674345, 0.30000044875568443, -0.09867003254982368, -0.2818454638606835, -0.12104353126524155, -0.16465859232487173, 0.13741272093125775, 0.2945987872493929, 0.13894867625622664, 0.23037092714928245, -0.07078625572409582, 0.08340063837181769, 0.22322562109384575, 0.3574517124526822, -0.11268975950635272,-0.04977623833253638, 0.01642898823084157, 0.18862161435042726, -0.5334214784692743, 0.014759044757193661, 0.3680519398840982, 0.28150268359023944, 0.5520106670499774, 0.4906508726274032, -0.09220187073799238, 0.2729626577319537, 0.5760868426882666, 0.16126425303444386, 0.2798640376392894, -0.2970302857457692, -0.4759670854798897, 0.4042983449784513, 0.6891805240012313, 0.587056527928376, -0.40155678749811585, 0.6369214993872552, -0.5876573550235041, -0.03501934974838357, -0.05093446942513047, -0.39712344260913024, -0.42642314771964795, 0.5083636592412561, -0.07602880933194309, 0.5888554843933193, -0.34433637211332824, 0.4952321725877591, 0.4875922343753649, 0.3181799661089931, -0.5728919173971239, -0.5013346584562525, -0.5237614602925892, 0.19766365684630255, 0.1887996853619211, -0.5357331654574832, 0.5288547752862219, -0.11824143830339463, 0.5285150254722258, 0.551552846929169, 0.08722730312759208, -0.42550545785136856, 0.4918379532441811, 0.23762394766565828, -0.5469688523291781, -0.34840065905696654, 0.33972805494089225]
pt.plot(X, Y, linestyle='', marker='.')
# pt.plot(X1, Y1, linestyle='', marker='.')
pt.show()
'''
