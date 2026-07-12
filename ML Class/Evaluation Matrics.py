#These are two 1. classification metrics and Regression metric.


#1. Calssification metrics

#Accuracy
#Formula = TP + TN / TP+TN+FN+FP
#confusion Metrics 
#True positive - CORRECTLY PREDICTED POSTIVE CASES
#true Negative - CORRECTLY PREDICTED NEGATIVE CASES
#False Positive - INCORERECTLY PREDICTED POSITIVE CASES
#False Negative  - INCORRECTLY PREDICTED NEGATIVE  CASES

from sklearn.metrics import accuracy_score

y_true = [1,1,0,1,0]
y_pred = [1,1,0,0,0]

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy:  {accuracy}")


#precision 
# TP / TP+FP

from sklearn.metrics import precision_score

y_true = [1,1,1,0,1,1]
y_pred = [1,1,0,1,0,1]

precision = precision_score(y_true , y_pred)
print(f"Precision : {precision}")


#recall 
#TP / Tp+ tn

from sklearn.metrics import recall_score

y_true = [1,1,0,0,1]
y_pred = [1,1,1,0,0]

Recall = recall_score(y_true, y_pred)
print(f"Recall : {Recall}")


#F1 Score 
#when ever our dataset contains fp and fn  and for balancing the dataset we can use the f1 score
#f1 = 2*precision *Recall / Precision + Recall
#| F1 SCORE  | MEANING       |
#| 1.0       | Perfect Model |
#| 0.8 - 0.9 | very good     |
#| 0.6 - 0.8 | Good          |
#|0.5 - 0.6  | Average       |
#|Below 0.5  | Poor          |

from sklearn.metrics import f1_score

y_true = [1,1,0,0,1]
y_pred = [1,1,1,0,0]

f_score =  f1_score(y_true, y_pred)
print(f"F1_score: {f_score}")


# Metric     |  Question It Answers                            |
# Accuracy   | How often is the Model correct?                 |
# Precision  | When the Model says YES, how often is it right? |
# Recall     | Out of All YES cases, how many did it find?     |
# F1 Score   | How Will Does it balance precision and Recall?  |
 
#Roc Curve
#receiver operating Characteristic Curve
# True positive rate(TRP) on the y_axis
# False positive rate (FPR) on the x_axis 
from sklearn.metrics import roc_curve,roc_auc_score

y_true = [0,0,1,1]
y_score = [0.1, 0.4, 0.35, 0.8]

auc = roc_auc_score(y_true , y_score)
print(f"AUC : {auc}")




#Evaluation metrics for regression

#MAE = mean absoulte error
#MSE = mean squared error
#RMSE = root mean square error

#R-squared (Coefficient of Determination)
#R^2 = 1:     perfect prediction
#R^2 = 0:     model performs no better than predicting the mean
#R^2 < 0:    model performs worse than predcting the mean


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)
y_true= [10,15,20,25]
y_pred = [8,18, 19, 22]

print("MAE: ", mean_absolute_error(y_true, y_pred))
print("MSE: ", mean_squared_error(y_true, y_pred))
print("RMSE: ", root_mean_squared_error(y_true, y_pred))
print("R Score: ", r2_score(y_true, y_pred))