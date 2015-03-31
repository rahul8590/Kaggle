import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression , BayesianRidge , SGDRegressor
from sklearn.feature_selection import VarianceThreshold


data = np.load('train.npy'); 
data_test = np.load('test_distribute.npy')
Y_train = data[:, 0] 
X_train = data[:, 1:] 
X_test = data_test[:, 1:]

std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.fit_transform(X_test)

#X_train = VarianceThreshold().fit_transform(X_train)
#X_test = VarianceThreshold().fit_transform(X_test)

'''
clf = LogisticRegression(C=8.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l1', random_state=None, tol=0.0001)
'''
'''
clf = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
       fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=100,
       normalize=False, tol=0.001, verbose=False)
'''
clf = SGDRegressor(alpha=0.0001, epsilon=0.1, eta0=0.01, fit_intercept=True,
       l1_ratio=0.15, learning_rate='invscaling', loss='squared_loss',
       n_iter=5, penalty='l1', power_t=0.25, random_state=None,
       shuffle=False, verbose=0, warm_start=False)


clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

fout = open('sgd_crimes_v3.kaggle', 'w')
fout.write("ID,Target\n");
for i in range(Y_pred.shape[0]):
      fout.write("\n" + str(float(i+1)) + "," + str(float(Y_pred[i]))) ## str(int(Y_pred[i]
fout.close()



