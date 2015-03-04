import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn import tree


import timeit

data = np.load('/rahul_extra/kaggle_blackbox/train.npy'); 
data_test = np.load('/rahul_extra/kaggle_blackbox/test_distribute.npy')
Y_train = data[:, 0] 
X_train = data[:, 1:] 
X_test = data_test[:, 1:]

#std_scaler = StandardScaler()
#X_train = std_scaler.fit_transform(X_train)
#X_test = std_scaler.fit_transform(X_test)

#X_train = Normalizer().fit_transform(X_train)
#X_test = Normalizer().fit_transform(X_test)

#clf = LogisticRegression(penalty='l2')  ## logistic 

#BEst so far clf = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, penalty='l1', random_state=None, tol=0.0001)
#clf = LogisticRegression(C=0.06, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

#clf = LogisticRegression(C=0.05, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, penalty='l1', random_state=None, tol=0.0001)

clf = tree.DecisionTreeClassifier()

def fit():
	clf.fit(X_train, Y_train)

ftime = timeit.timeit(fit,number=1)
print "fitting time",ftime*100


Y_pred = clf.predict(X_test)


def predict():
	Y_pred = clf.predict(X_test)

ptime = timeit.timeit(predict,number=1)
print "predicting time",ptime*100
	

#fittime = timeit.timeit(fit,number=10)
#print "prediction time ",fittime*10


fout = open('dt_default_blackbox.kaggle', 'w')
fout.write("ID,Category\n");
for i in range(Y_pred.shape[0]):
      fout.write("\n" + str(i+1) + "," + str(int(Y_pred[i]))) ## str(int(Y_pred[i]
## now make kaggle submission Y_pred 
fout.close()
