import numpy as np
#from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge, Lasso , lasso_path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier


# leafy-audio-87805

data = np.load('train.npy'); 
y = data[:, 0] 
X = data[:, 1:] 




# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

#Full Datasets taken into consideration
#Xf_train = X

#---- Uncomment the lines to choosing the apt Scaler -----------

#X_train = VarianceThreshold().fit_transform(X_train)


#X_train = Normalizer().fit_transform(X_train)
#X_test = Normalizer().fit_transform(X_test)
#------------------------------------------------------------------

#clf = ExtraTreesClassifier()
#X_train = clf.fit(X_train, y_train).transform(X_train)


std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.fit_transform(X_test)


print "X train shape",X_train.shape
print "X_test share",X_test.shape

#Hyper Parameters for various Classifiers----------------------------

# Set the parameters by cross-validation SVM/R
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1,10,50, 100]}]

#tuned_parameters = [{'n_estimators': [50], 'max_depth': [3,None],'learning_rate' : [0.1,0.3,0.5], 'min_samples_leaf' : [3,5,8] },
#                    {'n_estimators': [75], 'max_depth' :[5,None] ,'learning_rate' : [0.3,0.5,0.8], 'min_samples_leaf' : [3,5,8]
#                    }]

#Parameters for SGD Regressor
#tuned_parameters = [{'penalty': ['l1','l2','elasticnet'],'loss' :['squared_loss','huber'] }]

#Parameters for BayianRidge
#tuned_parameters = [{'n_iter' : [300,400,500]}]

#Parameters for RandomForestREgressor
#tuned_parameters = [{'n_estimators':[70,80,90]}]

#Parameters for Lasso
#tuned_parameters = [{'alpha':[0.006,0.007,0.008,0.009,0.010]}]

#Parameters for Lasso path
tuned_parameters = [{'eps':[0.001,0.002,0.003]}]

# C values in right range 0.06,0.065,0.07,0.075,0.071,0.078
#-------------------------------------------------------------------------

#tuned_parameters = [{'penalty': ['l1','l2'] , 'C':[5.0,6.0,7.0,8.0,9.0]}]

'''
Possible score options are
['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 
'log_loss', 'mean_squared_error', 'precision', 'r2', 'recall', 'roc_auc']
'''


scores = ['mean_squared_error']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(lasso_path(X_train,y_train), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
        #clf_whole = LogisticRegression().set_params(**params)
        #clf_whole.fit(Xf_train, y)
        #Yf_pred = clf_whole.predict(Xf_train)
        #print accuracy_score(y,Yf_pred)
