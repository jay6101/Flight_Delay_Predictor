#Assignment-2_GNR652
#18D070050
#import_libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
################## Function_Definitions #########################################
def CostFunction(X,Y,B):
    m = len(X)
    z=-1*np.dot(X,B)
    h=1/(1+np.exp(z))
    cost = Y*(np.log(h))+ (1-Y)*(np.log(1-h+1e-10))
    J= - (1/m)*(cost.sum())
    return J
    
def Logistic_training_model(X,Y,learning_rate,iterations):
    
    B = np.zeros([(len(X[0])),1])
    m = len(X)
    costHistory = np.zeros([iterations,1])
    for i in range(iterations):
        costHistory[i]= CostFunction(X,Y,B)
        z=-1*np.dot(X,B)
        h=1/(1+np.exp(z)) 
        gradient = ((X.transpose())@(Y-h))/m
        B=B+learning_rate*gradient
        
    return costHistory,B

def standardize(x):
    mean = np.sum(x)/len(x)
    stdev = np.sqrt(np.sum((x-mean)*(x-mean))/len(x))
    m = (x-mean)/stdev
    return m
   
########################################################################################	
dataset = pd.read_csv('FlightDelays.csv')


##########################PRE-PROCESSING DATA###########################################

dataset.drop(['FL_DATE'],axis=1,inplace=True)

label_encoder = LabelEncoder()
dataset['Flight Status'] = label_encoder.fit_transform(dataset['Flight Status'])

dataset = pd.get_dummies(data=dataset, columns = ['CARRIER','DEST','ORIGIN','DAY_WEEK','FL_NUM','DAY_OF_MONTH','TAIL_NUM'])


########################################################################################

Y = (dataset['Flight Status'].values).reshape(2201,1)
dataset.drop(['Flight Status'],axis=1,inplace=True)
X=dataset[:].values.astype(float)
X0 = np.ones([len(X),1])

X[:,0] = ((X[:,0]%100) + (((X[:,0]-(X[:,0]%100))/100)*60))

X[:,1] = ((X[:,1]%100) + (((X[:,1]-(X[:,1]%100))/100)*60))

X= np.hstack((X0,X))
######################## Splitting_Data #################################################
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4)
X_train[:,1] = standardize(X_train[:,1])
X_train[:,2] = standardize(X_train[:,2])
X_train[:,3] = standardize(X_train[:,3])#Distance


X_test[:,3] = standardize(X_test[:,3])#Distance
X_test[:,2] = standardize(X_test[:,2])
X_test[:,1] = standardize(X_test[:,1])

######################## Training_the_Model ##############################################

B = np.zeros([(len(X_train[0])),1])
icos = CostFunction(X_train,y_train,B)

costH,newB = Logistic_training_model(X_train,y_train,1,30000)

plt.plot(costH)
plt.show()

######################## Classification ##################################################
Z=-1*np.dot(X_test,newB)
h=1/(1+np.exp(Z))
for i in range(h.shape[0]):
    if h[i]>0.5:
        h[i] = 1
    else:
        h[i] = 0
######################## Accuracy_Calculation ###########################################
TP=FN=FP=TN=0
   
for i in range(len(y_test)) :
    if (y_test[i]==1):
        if (h[i]==1):
            TP=TP+1
        else:
            FN=FN+1
    else :
        if(h[i]==0):
            TN=TN+1
        else:
            FP=FP+1

Accuracy=(TP+TN)/(TP+TN+FP+FN)
error=(FP+FN)/(TP+TN+FP+FN)

print("Accuracy = ",Accuracy)
print("Error = ",error)
    





















