import numpy as np 
import matplotlib.pyplot as plt 


class ROC:
	# def __init__(self):
	def Roc_Cur(self,y_test,y_pred_prob):
		y_true = y_test
		# Sort instances by predicted probabilities
		order = np.argsort(y_pred_prob)[::-1]
		y_true = y_true[order]
		y_pred_prob = y_pred_prob[order]

		# Calculate TPR and FPR for each threshold
		tpr_list = []
		fpr_list = []
		thresholds = np.unique(y_pred_prob)
		thresholds = np.concatenate((thresholds, [thresholds[-1] + 1]))
		for threshold in thresholds:
		    y_pred = (y_pred_prob >= threshold).astype(int)
		    tp = np.sum(y_true * y_pred)
		    tn = np.sum((1 - y_true) * (1 - y_pred))
		    fp = np.sum((1 - y_true) * y_pred)
		    fn = np.sum(y_true * (1 - y_pred))
		    tpr = tp / (tp + fn)
		    fpr = fp / (fp + tn)
		    tpr_list.append(tpr)
		    fpr_list.append(fpr)

		# Plot ROC curve
		plt.plot(fpr_list, tpr_list)
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC Curve')
		plt.show()

		# Calculate ROC score
		roc_score = np.trapz(tpr_list, fpr_list)
		print('ROC score:', roc_score)