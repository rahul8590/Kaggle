import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression



data = np.load('train.npy'); 
data_test = np.load('test_distribute.npy')
Y_train = data[:, 0] 
X_train = data[:, 1:] 
X_test = data_test[:, 1:]

std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.fit_transform(X_test)

clf = LogisticRegression(C=5.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l1', random_state=None, tol=0.0001)

clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

fout = open('lr_crimes_v1.kaggle', 'w')
fout.write("ID,Target\n");
for i in range(Y_pred.shape[0]):
      fout.write("\n'" + str(float(i+1)) + "'," + str(float(Y_pred[i]))) ## str(int(Y_pred[i]
fout.close()



