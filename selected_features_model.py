#Assignment-2_GNR652
#18D070050
#import_libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#################### Function Definitions ########################
def CostFunction(X,Y,B):
    m = len(X)
    z=-1*np.dot(X,B)
    h=1/(1+np.exp(z))
    cost = Y*(np.log(h))+ (1-Y)*(np.log(1-h+1e-5))
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

def cor(x,y):
	mux = (np.sum(x))/len(x)
	muy = (np.sum(y))/len(y)
	num = np.sum((x-mux)*(y-muy))
	denx = np.sum((x-mux)*(x-mux))
	deny = np.sum((y-muy)*(y -muy))
	stdx = np.sqrt(denx)+1e-20
	stdy= np.sqrt(deny)+1e-20
	ret = num/np.sqrt(denx*deny)
	return abs(ret)
    
	
dataset = pd.read_csv('FlightDelays.csv')


####################PRE-PROCESSING DATA#############################

dataset.drop(['FL_DATE','TAIL_NUM'],axis=1,inplace=True)

label_encoder = LabelEncoder()
dataset['Flight Status'] = label_encoder.fit_transform(dataset['Flight Status'])

dataset = pd.get_dummies(data=dataset, columns = ['CARRIER','DEST','ORIGIN','DAY_WEEK','FL_NUM','DAY_OF_MONTH'])


####################################################################

Y = (dataset['Flight Status'].values).reshape(2201,1)
dataset.drop(['Flight Status'],axis=1,inplace=True)
X=dataset[:].values.astype(float)
X0 = np.ones([len(X),1])

X[:,0] = ((X[:,0]%100) + (((X[:,0]-(X[:,0]%100))/100)*60))
X[:,0] = standardize(X[:,0])
X[:,1] = ((X[:,1]%100) + (((X[:,1]-(X[:,1]%100))/100)*60))
X[:,1] = standardize(X[:,1])
################# Feature_Selection ######################################
corr=np.zeros(len(X[0]))
for i in range(len(X[0])):
    corr[i]= cor((X[:,i]).reshape(2201,1),Y)
ypos=np.arange(len(corr))

for i in range(len(X[0])):
    if abs(corr[i]) <= 0.1:
        corr[i]=0
    else :
        corr[i]=1
X_del=np.ones([2201,1])

for i in range(len(X[0])):
    if corr[i]==1:
        X_del=np.hstack((X_del,(X[:,i]).reshape(2201,1)))



########################## Splitting_Dataset ###############################
X_train, X_test, y_train, y_test = train_test_split(X_del, Y, test_size = 0.4)

######################### Training_of_Model ##################################

B = np.zeros([(len(X_train[0])),1])
icos = CostFunction(X_train,y_train,B)

costH,newB = Logistic_training_model(X_train,y_train,1,30000)

plt.plot(costH)
plt.show()
######################### Classification ######################################
Z=-1*np.dot(X_test,newB)
h=1/(1+np.exp(Z))
for i in range(h.shape[0]):
    if h[i]>0.5:
        h[i] = 1
    else:
        h[i] = 0


######################### Accuracy_calculation ################################
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
    





















