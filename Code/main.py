import numpy as np
from NeuronNetwork import NeuralNetwork
from Euler_SIRD import EulerSIRD
import matplotlib.pyplot as plt

# Read data from csv file
my_data = np.genfromtxt('hubei.csv', delimiter=',', dtype= None)
input_train = my_data[:,0:60]
input_train = input_train.astype(np.float64)
output_train = my_data[:,0:61]
output_train = output_train.astype(np.float64)

# Traing with dataset
neuron = NeuralNetwork(input_train.T,output_train.T,58500000)

X_test = my_data[:,0:60]
predict_alpha = neuron.predict(X_test.T)

#Test with model
U_predict = []
count = 0
for i in range(60):
    if(count == 0):
        data = my_data[:,i].T
        U_0 = [58500000.0-data[0]-data[1]-data[2],data[0],data[1],data[2]]
        print (U_0)
    else:
        U_0 = U_predict[-1]
    predict_alpha_i = predict_alpha[i]
    m = EulerSIRD(predict_alpha_i[0],predict_alpha_i[1],predict_alpha_i[2],U_0)
    U_x = m.ode_FE()
    U_predict.append(U_x)
    count += 1

# Draw graph
time = np.arange(60)
plt.xlabel('Time')
plt.ylabel('Population')
plt.plot(time,np.array(U_predict)[:,1].T, 'r--')
plt.plot(time,my_data[0,0:60].T, 'r')
plt.plot(time,np.array(U_predict)[:,2].T, 'b--')
plt.plot(time,my_data[1,0:60].T, 'b')
plt.plot(time,np.array(U_predict)[:,3].T, 'g--')
plt.plot(time,my_data[2,0:60].T, 'g')
plt.legend(['Prediction I','Real data I','Prediction R','Real data R','Prediction D','Real data D'], prop={'size': 8}, loc='upper left', fancybox=True, shadow=True)
plt.show()
