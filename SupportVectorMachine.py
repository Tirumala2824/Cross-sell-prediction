import numpy as np


class SupportVectorMachine():

	def __init__(self,learning_rate=0.2,iterations=1000,lambda_parametrs=0.1):

		self.learning_rate=learning_rate
		self.iterations=iterations
		self.lambda_parametrs=lambda_parametrs

	def fit(self,X,y):
		# self.m is a no of  column in a dataset
		#self.n is a no  of rows in a dataset
		self.m,self.n = X.shape

		self.w = np.zeros(self.n)
		self.b = 0

		self.x=X
		self.y=y

		for i in range(self.iterations):
			self.update_weights()

	# gradients (dw,db)
	def update_weights(self):

		y_label=np.where(self.y <=0,-1,1)

		for index,xi in enumerate(self.x):
			condition = y_label[index] * (np.dot(xi,self.w) - self.b) >=1
			if (condition == True):
				dw = 2* (self.lambda_parametrs)*self.w
				db=0
			else:
				dw = 2 * (self.lambda_parametrs) * (self.w - np.dot(xi,y_label[index]))
				db = y_label[index]

			self.w = self.w - self.learning_rate *dw 
			self.b = self.b - self.learning_rate * db

	def predict(self,X):

		prediction_output = np.dot(X,self.w) - self.b

		prediction_labels = np.sign(prediction_output)

		y_pred = np.where(prediction_labels <= -1,0,1)

		return y_pred,prediction_labels
	def predict_proba(self,X):
		prediction_output = np.dot(X,self.w) - self.b
		return prediction_output
		 



