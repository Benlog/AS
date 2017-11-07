import pandas as pd
import numpy as np

def read_libsvm(fname):
    with open(fname) as f:
        x,y  =list(),list()
        for l in f:
            line = l.strip().split(" ")
            y.append(int(line[0]))
            x.append([float(s.split(":")[1]) for s in line[1:]])
    return np.array(x),np.array(y)

class Loss:
    def forward(self, yPred, y):
        pass

    def backward(self, yPred, y):
        pass

class Module:
    def forward(self, x):
        pass

    def backward(self, x, delta):
        pass

    def update(self):
        pass

def gradSto(m, l, x, y, n = 10000, minErr = 0):
    ''' x in size N*arg'''
    for i in range(n):
        j = np.random.randint(0,len(x))
        yPred = m.forward(x[j])
        err = l.forward(yPred, y[j])
        delta = l.backward(yPred, y[j])
        m.backward(x, delta)
        m.update()
        if (err <= minErr):
            break

class SquareLoss:
    def forward(self, yPred, y):
        return np.mean((yPred - y) * (yPred - y))

    def backward(self, yPred, y):
        return 2 * yPred - 2 * y
        
class Linear:
    def __init__(self, e = 0.1):
        self.ini = False
        self.eps = e
    
    def init(self, x):
        self.param = np.random.random(x.shape[1]+1)
        self.grad = np.zeros(x.shape[1]+1)
        self.ini = True

    def forward(self, x):
        if self.ini == False :
            self.init(x)
        x.reshape((-1,self.param.shape[0]-1))
        return np.sum(np.concatenate((np.ones((x.shape[0],1)), x), axis = 1) * self.param, 1)
    
    def backward(self, x, theta):
        g = theta * x
        self.grad = np.concatenate(self.grad, g)
        param = param - self.eps * g

m = Linear()
l = SquareLoss()
x,y = read_libsvm("./breast-cancer_scale")
m.forward(x)
gradSto(m,l,x,y)