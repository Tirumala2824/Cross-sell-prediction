import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns 
from ROC_AUC import ROC 

from LogisticRegreesion import Logisticregression


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

#print(type(y1_train))


x_train=x1_train.to_numpy()
y_train=y1_train.to_numpy()
print(y_train)
x_test=x1_test.to_numpy()
y_test=y1_test.to_numpy()

#print(type(x_test))




def Data_visualization(x_train):
	plt.figure(figsize=(10,6))
	plt.scatter(x_train[:, 0], x_train[:, 5], c=x_train[:, 2], cmap='winter')
	plt.show()
# Data_visualization(x_train)
###############################################
###############################################
###############################################
def al_implement():
	LR = Logisticregression()

	LR.fit(x_train,y_train)
	y_predicts=LR.predict(x_test)
	print(f"y_predictions : {y_predicts}")
	return y_predicts

y_predss=al_implement()
def pobablilites_y():
	LR1 = Logisticregression()

	LR1.fit(x_train,y_train)
	y_pred_proba = LR1.predict_probability(x_test)
	print(f"y_predictions_Probablity:{y_pred_proba}")
	return y_pred_proba
y_pred_prob=pobablilites_y()
print(f"predict probability : {y_pred_prob} ")

def Accuracy(y_pred):
	correct = 0
	for i in range(y_test.shape[0]):
	    # print(y_test[i])
	    # print(y_pred[i])
	    if y_test[i] == y_pred[i]:
	        correct += 1
	accuracy = correct / len(y_test)

	print('Accuracy:', accuracy)
Accuracy(y_predss)

LR_ROC = ROC()
LR_ROC.Roc_Cur(y_test,y_pred_prob[:,1])

############################################
################################################
#################################################
#################################################

print(f"length of y_shape: {y_test.shape[0]}")
print(f"length of y_shape: {y_train.shape[0]}")


# Test function




################################################
################################################
#################################################

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score,roc_curve

Sk_LR=LogisticRegression(solver='lbfgs',max_iter=10000)

Sk_LR.fit(x_train,y_train)

acc_score = Sk_LR.score(x_test,y_test)

pred = Sk_LR.predict(x_test)
print(f"ypredicts for sklearn : {pred}")
print(f"Accuracy in Sklearn : {acc_score}")
sky_test_probs = Sk_LR.predict_proba(x_test)[:,1]
print(f"Sk learn probability : {sky_test_probs}")



Sk_roc=ROC()

Sk_roc.Roc_Cur(y_test,sky_test_probs)



#################################################
#################################################
#################################################


# y_true = y_test
# # Sort instances by predicted probabilities
# order = np.argsort(y_pred_prob)[::-1]
# y_true = y_true[order]
# y_pred_prob = y_pred_prob[order]

# # Calculate TPR and FPR for each threshold
# tpr_list = []
# fpr_list = []
# thresholds = np.unique(y_pred_prob)
# thresholds = np.concatenate((thresholds, [thresholds[-1] + 1]))
# for threshold in thresholds:
#     y_pred = (y_pred_prob >= threshold).astype(int)
#     tp = np.sum(y_true * y_pred)
#     tn = np.sum((1 - y_true) * (1 - y_pred))
#     fp = np.sum((1 - y_true) * y_pred)
#     fn = np.sum(y_true * (1 - y_pred))
#     tpr = tp / (tp + fn)
#     fpr = fp / (fp + tn)
#     tpr_list.append(tpr)
#     fpr_list.append(fpr)

# # Plot ROC curve
# plt.plot(fpr_list, tpr_list)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.show()

# # Calculate ROC score
# roc_score = np.trapz(tpr_list, fpr_list)
# print('ROC score:', roc_score)




