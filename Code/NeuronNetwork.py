import numpy as np
from numpy import exp, array, random, dot
from keras.layers import Dense 
from keras.models import Sequential,Model
from keras.layers import Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import math

class NeuralNetwork:
    def __init__(self, x, y, population):
        self.x = x
        self.y = y
        self.population = population
        self.maxIc = 0

    def predict(self,test):
        # Calculate output
        self.calOut()

        # Preprocess data
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_x.fit(self.x)
        xscale=scaler_x.transform(self.x)
        scaler_y.fit(self.out)
        yscale=scaler_y.transform(self.out)

        #Config model
        model = Sequential()
        model.add(Dense(64,activation='sigmoid',input_dim=3))
        model.add(Dense(32,activation='sigmoid'))
        model.add(Dense(16,activation='sigmoid'))
        model.add(Dense(3,activation='sigmoid'))
     
        opt = Adam(lr=0.01)
        model.compile(loss=self.customLoss, optimizer=opt, metrics=['mse','mae'])
        model.fit(xscale, yscale, epochs=500,batch_size=1,validation_split=0.2)
        print (model.evaluate(self.x, self.out))

        # Predict sample
        ynew= model.predict(xscale)
        ynew = scaler_y.inverse_transform(ynew)

        # Draw graph
        time = np.arange(60)
        plt.xlabel('Time')
        plt.ylabel('alpha')
        plt.plot(time,self.out[:,0], 'r')
        plt.plot(time,ynew[:,0], 'r--')
        plt.legend(['Real data beta','Prediction beta'], prop={'size': 8}, loc='upper left', fancybox=True, shadow=True)
        plt.show()
        plt.plot(time,self.out[:,1], 'b')
        plt.plot(time,ynew[:,1], 'b--')
        plt.legend(['Real data gamma','Prediction gamma'], prop={'size': 8}, loc='upper left', fancybox=True, shadow=True)
        plt.show()
        plt.plot(time,self.out[:,2], 'g')
        plt.plot(time,ynew[:,2], 'g--')
        plt.legend(['Real data mu','Prediction mu'], prop={'size': 8}, loc='upper left', fancybox=True, shadow=True)
        plt.show()
        plt.plot(time,np.multiply(np.divide(self.out[:,0],self.out[:,1]),58500000), 'y')
        plt.plot(time,np.multiply(np.divide(ynew[:,0],ynew[:,1]),58500000), 'y--')
        plt.legend(['Real data R0','Prediction R0'], prop={'size': 8}, loc='upper left', fancybox=True, shadow=True)
        plt.show()

        return ynew

    def customLoss(self, yTrue,yPred):
        Ed1 = K.sum(K.square(K.log(yTrue + 1) - K.log(yPred + 1))) 
        Ed2 = 0.01*math.log(self.maxIc)/self.maxIc* K.sum(K.square(yTrue - yPred))
        return Ed1 + Ed2

    def calOut(self):
        out = []
        for i in range(self.y.shape[0]):
            if(i<self.y.shape[0]-1):
                if(self.y[i,0] > self.maxIc): 
                    self.maxIc = self.y[i,0]

                if (self.y[i,0] != 0):
                    mu = (self.y[i+1,2] - self.y[i,2])/self.y[i,0]
                else: 
                    mu = 0
                    
                if (self.y[i,0] != 0):
                    gama = (self.y[i+1,1] - self.y[i,1])/self.y[i,0]
                else:
                    gama = 0

                Si = self.population-(self.y[i,0] + self.y[i,1] + self.y[i,2])
                Si_1 = self.population-(self.y[i+1,0] + self.y[i+1,1] + self.y[i+1,2])

                if(self.y[i,0] != 0):
                    beta = (Si - Si_1)/(Si*self.y[i,0])
                else:
                    beta = 0
                out.append([beta, gama, mu])

        self.out = np.array(out)

