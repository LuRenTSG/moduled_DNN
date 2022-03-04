#import lib
from cProfile import label
import math
import numpy as np
import copy
import random as rd
from scipy import special

#define function
def sigmoid(x):
    if type(x)==np.ndarray:
        ans=np.zeros(len(x))
        for i in range(len(x)):
            if x[i]<=-700:
                ans[i]=0
            else:
                ans[i]=1/(1+math.exp(-x[i]))
        return ans
    if type(x)==list:
        ans=[]
        for i in range(len(x)):
            ans.append(1/(1+math.exp(-x[i])))
        return ans
    return 1/(1+math.exp(-x))
        
def s_prime(x):
    if type(x)==np.ndarray:
        ans=np.zeros(len(x))
        for i in range(len(x)):
            ans[i]=math.exp(-x[i])/((1+math.exp(-x[i]))**2)
        return ans
    if type(x)==list:
        ans=[]
        for i in range(len(x)):
            ans.append(math.exp(-x[i])/((1+math.exp(-x[i]))**2))
        return ans
    return math.exp(-x)/((1+math.exp(-x))**2)

def normal_random():
    return math.sqrt(2)*special.erfinv(2*rd.random()-1)




class network:
    def __init__(self,config):
        self.value=[]
        self.weight=[]
        self.bias=[]
        self.error=0
        self.alpha=0.1
        self.deA=None
        self.deB=None
        self.config=config
        self.output=None
        #config not a list, print error and return
        if type(config)!=list:
            print('config need to be a list')
            return
        #build network
        for i in range(len(config)):
            self.value.append(np.array([0.0]*config[i]))
            for j in range(len(self.value[i])):
                self.value[i][j]=normal_random()
        #weight
        for i in range(len(config)-1):
            self.weight.append(np.array([[0.0]*config[i+1]]*config[i]))
            for j in range(len(self.weight[i])):
                for k in range(len(self.weight[i][j])):
                    self.weight[i][j][k]=normal_random()
        #bias
        for i in range(len(config)-1):
            self.bias.append(np.array([0.0]*config[i+1]))
            for j in range(len(self.bias[i])):
                self.bias[i][j]=normal_random()
    #predict
    def predict(self,input_data):
        #input data not array, print error and return
        if type(input_data)!=np.ndarray:
            print('input not array')
            return
        #shape difference, print error and return
        if input_data.shape!=self.value[0].shape:
            print('shape error')
            return
        self.value[0]=input_data
        for i in range(1,len(self.value)):
            self.value[i]=sigmoid(np.dot(self.value[i-1],self.weight[i-1])+self.bias[i-1])
        self.output=self.value[-1]
    #caculate error
    def getError(self,label):
        #input data not array, print error and return
        if type(label)!=np.ndarray:
            print('label not array')
            return
        #shape difference, print error and return
        if label.shape!=self.value[-1].shape:
            print('shape error')
            return
        self.error=(np.sum((self.value[-1]-label)**2))
    #upload the weight of network
    def upload(self):
        for i in range(len(self.weight)):
            self.weight[i]-=self.deA[i]*self.alpha
        for i in range(len(self.bias)):
            self.bias[i]-=self.deB[i]*self.alpha
    #caculate step for weight and bias
    def getStep(self,train,label):
        deA=copy.deepcopy(self.weight)
        deB=copy.deepcopy(self.bias)
        deValue=copy.deepcopy(self.value)
        
        for i in range(len(deValue[-1])):
            deValue[-1][i]=2*(self.output[i]-label[i])
        
        # de/dy1=(de/dz1*dz1/dy1+de/dz2*dz2/dy1+...)
        # dz1/dy1=s'(wa11*y1+wa21*y2+wa31*y3+..ba1)*wa11
        for i in range(len(deValue)-2,-1,-1):
            for j in range(len(deValue[i])):
                ans=0
                for k in range(len(self.value[i+1])):
                    tmp=0
                    for l in range(len(self.value)):
                        tmp+=(self.weight[i][l][k]*self.value[i][l])
                    tmp+=self.bias[i][k]
                    tmp=s_prime(tmp)
                    tmp*=(self.weight[i][j][k]*deValue[i+1][k])
                    ans+=tmp
                deValue[i][j]=ans
        
        # a:x->y
        # de/dwabc=de/dy*dy/dwabc
        for i in range(len(deA)):
            for j in range(len(deA[i])):
                for k in range(len(deA[i][j])):
                    tmp=0
                    for l in range(len(self.value[i])):
                        tmp+=(self.value[i][l]*self.weight[i][l][k])
                    tmp=s_prime(tmp)
                    deA[i][j][k]=deValue[i+1][k]*tmp*self.value[i][j]


        for i in range(len(deB)):
            for j in range(len(deB[i])):
                tmp=0
                for k in range(len(self.value[i])):
                    tmp+=(self.value[i][k]*self.weight[i][k][j])
                tmp=s_prime(tmp)
                deB[i][j]=deValue[i+1][j]*tmp

        self.deA=deA
        self.deB=deB




    def train(self,train_data,train_label,epoch):
        if type(train_data)!=list and type(train_label)!=list:
            print('train data or label have to be list')
            return 
        if len(train_data)!=len(train_label):
            print('length of data and label have to be the same')
            return
        for j in range(epoch):
            total_error=0
            for i in range(len(train_data)):
                print('\r',j+1,'-',i+1,end='')
                self.predict(train_data[i])
                self.getError(train_label[i])
                total_error+=self.error
                self.getStep(train_data[i],train_label[i])
                self.upload()
            print(' average error: '+str(total_error/len(train_data)))
    def printResult(self):
        print(self.output)