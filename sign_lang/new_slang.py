import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

data = np.load('train.npy'); 
data_test = np.load('test_distribute.npy')
Y_train = data[:, 0] 
X_train = data[:, 1:] 
X_test = data_test[:, 1:]

std_scaler = MinMaxScaler()
X_train = std_scaler.fit_transform(X_train)

clf = GradientBoostingClassifier()  ## logistic ,
clf.fit(X_train, Y_train)
X_test = std_scaler.fit_transform(X_test)
Y_pred = clf.predict(X_test)

fout = open('rahul_ram-slang_gradient.kaggle', 'w')
fout.write("ID,Category\n");
for i in range(Y_pred.shape[0]):
      fout.write("\n" + str(i+1) + "," + str(int(Y_pred[i]))) ## str(int(Y_pred[i]
## now make kaggle submission Y_pred 
fout.close()

