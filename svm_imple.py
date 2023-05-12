from SupportVectorMachine import SupportVectorMachine

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns 
from ROC_AUC import ROC 

# from LogisticRegreesion import Logisticregression


'''
Trainsize is 70%
data_train = 266777
data_test = 114333
''' 

def DataPath():

	data_train =pd.read_csv(r"train.csv",skiprows=range(266778,381110))
	#data_train=pd.DataFrame(df_train)
	data_count=data_train.shape[0]
	train_size = data_count*(.7)
	data_test = pd.read_csv(r"train.csv",skiprows=range(1,266777))
	#data_test = pd.DataFrame(df_test)

	gender_map={'Male':0,'Female':1}
	data_train['Gender']=data_train['Gender'].map(gender_map)
	#print(data_train['Gender'])
	vehicle_age_map={'1-2 Year':0,'< 1 Year':1,'> 2 Years':2}
	data_train['Vehicle_Age']=data_train['Vehicle_Age'].map(vehicle_age_map)
	Vehicle_Damage_map={'Yes':0,'No':1}
	data_train['Vehicle_Damage']=data_train['Vehicle_Damage'].map(Vehicle_Damage_map)


	gender_map={'Male':0,'Female':1}
	data_test['Gender']=data_test['Gender'].map(gender_map)
	#print(data_train['Gender'])
	vehicle_age_map={'1-2 Year':0,'< 1 Year':1,'> 2 Years':2}
	data_test['Vehicle_Age']=data_test['Vehicle_Age'].map(vehicle_age_map)
	Vehicle_Damage_map={'Yes':0,'No':1}
	data_test['Vehicle_Damage']=data_test['Vehicle_Damage'].map(Vehicle_Damage_map)

	

	X_train = data_train[['Gender', 'Age', 'Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel','Vintage']]
	Y_train=data_train['Response']
	X_test=data_test[['Gender','Age','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel','Vintage']]
	Y_test=data_test['Response']  
	# print(f'x_train: {X_train} \n Y_train : {Y_train} \n X_test : {X_test} \n Y_test: {Y_test}')

	return X_train,Y_train,X_test,Y_test

x1_train,y1_train,x1_test,y1_test=DataPath()

x_train=x1_train.to_numpy()
y_train=y1_train.to_numpy()
print(y_train)
x_test=x1_test.to_numpy()
y_test=y1_test.to_numpy()


svm1 = SupportVectorMachine()

svm1.fit(x_train,y_train)

y_pred ,y_pred_labels= svm1.predict(x_test)

print(f"t prediction : {y_pred}")



y_pred_proba=svm1.predict_proba(x_test)

print(f"y_prediction probability : {y_pred_proba}")

def Accuracy(y_pred):
	correct = 0
	for i in range(y_test.shape[0]):
	    # print(y_test[i])
	    # print(y_pred[i])
	    if y_test[i] == y_pred[i]:
	        correct += 1
	accuracy = correct / len(y_test)

	print('Accuracy:', accuracy)
Accuracy(y_pred)


from sklearn.svm import SVC

svm_sklearn = SVC(probability=True)

svm_sklearn.fit(x_train,y_train)
y_pred23=svm_sklearn.predict(x_test)


print(f"y_prediction_sklearn : {y_pred23}")
Accuracy(y_pred23)
# breakpoint()
sk_pred_proba = svm_sklearn.predict_proba(x_test)[:,1]
print(f"sk_pred_proba: {sk_pred_proba}")


svroc = ROC()

svroc.Roc_Cur(y_test,y_pred_proba)
skvm_roc=ROC()

skvm_roc.Roc_Cur(y_test,sk_pred_proba)
