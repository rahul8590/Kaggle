#!/bin/pyspark
import numpy as np
import time
from pyspark import SparkContext, SparkConf
import matplotlib.pyplot as plt
import math


basepath = '/rahul_extra/MachineLearning/HW3Data/'

def set_feature(view,f):
  x = float(view.split(' ')[2])
  return ((view.split(' ')[0],str(f)+".txt",view.split(' ')[1]),x)

def readfile():
	fr = range(6,23)
	tfile = sc.union([sc.textFile(basepath+str(f)+".txt")
		                .map(lambda view: set_feature(view,f)) 
		                .reduceByKey(lambda a, b: a+b)
		                for f in fr])
	return tfile

def map_group(pgroup):
	x = np.zeros(19)
	x[0] = 1
	value_list = pgroup[1]
	for val in value_list:
		fno = val[0].split('.')[0]
		x[int(fno)-5] = val[1]
	return x


def group(tfile):
	'''
	Should output the following 
	((u'cn', u'suability'),
		array([  1.,  26.,  38.,  16.,   0.,  33.,  41.,  35.,  13.,  24.,  39.,
         9.,  17.,   0.,  56.,  30.,   0.,   0.,   0.]))
	'''
	tgbr = tfile.map(lambda d: ((d[0][0],d[0][2]),[(d[0][1],d[1])])) \
				.reduceByKey(lambda p,q:p+q) \
				.map(lambda d: (d[0], map_group(d)))
	return tgbr

#The below function might soon be deprecated
def create_feature(tfile):
	collection = tfile.collect()
	d = {}
	for data in collection:
		if not d.has_key((data[0][0],data[0][2])):
			x = np.zeros(19)
			x[0] = 1
			fno = str(data[0][1]).split('.')[0]
			x[int(fno)-5] = data[1]
			d[(data[0][0],data[0][2])] = x
		else:
			x = d[(data[0][0],data[0][2])]
			fno = str(data[0][1]).split('.')[0]
			x[int(fno)-5] = data[1]
			d[(data[0][0],data[0][2])] = x
	return d

def get_target_info():
	tfile = sc.textFile(basepath+'23.txt')
	collection = tfile.map(lambda view: ((view.split(' ')[0],view.split(' ')[1]),float(view.split(' ')[2]))).reduceByKey(lambda a,b: a+b).collect()
	target = {}
	for data in collection:
		target[data[0]] = data[1]
	return target

#Deprecated Function
def ols(mat):
	'''
	mat => dcollec.values()
	mat contains the entire matrix [1,hr6-hr23]
	'''
	y = np.array(mat[0::,-1])
	x = mat[0:,:-1]
	#x = np.squeeze(np.asarray(mat[0:,:-1]))
	#A = np.vstack([x, np.ones(len(x))]).T
	
	#below equation solves the least square solution
	doutput = np.dot(np.linalg.inv(np.dot(x.T,x)),np.dot(x.T,y))
	print "the dot product output is ",doutput
	print doutput.shape

	#Solution 2
	#output = np.linalg.lstsq(x, y)[0]
	#print "the output for lstsq is ", output

	
#Function to take in the arrays and then 
# array([  1.,  26.,  38.,  16.,   0.,  33.,  41.,  35.,  13.,  24.,  39.,
#        9.,  17.,   0.,  56.,  30.,   0.,   0.,  58.])
#Calculate Sig (x,x^t)  and (xy)
def calc(x):
	pred = x[:18]
	sig1 = np.outer(pred, pred)
	sig2 = pred * x[18]
	return sig1

def calc_s2(x):
	pred = x[:18]
	sig2 = pred * x[18]
	return sig2


def predict(rdd,weight):
	pval = rdd.filter(lambda x: x[0][0] == 'en' and x[0][1]=='yahoo' ) \
	 		  .map(lambda d: np.dot(train_weight.T,d[1][:18])).collect()
	print "predicted value is",pval


#Function required to calculate RMSE
def update_yd(d):
	yd = np.dot(weight.T,d[1][:18])
	y = d[1][18]
	return (y - yd) * (y - yd)


#Adding the 23rd hour value to the last position in the array
def upd_t23(tgbr):
	key = tgbr[0]
	hrs = tgbr[1]
	if dtarget.has_key(key): #Checking if the hrs for 23rd exist
		hrs[18] = dtarget[key]
	return hrs


def main():
	tfile = readfile()

	dtarget = get_target_info()
	print "the data in 23rd file is", dtarget

	tgbr = group(tfile)

	tupdate = tgbr.map(lambda d: (d[0],upd_t23(d)))


	#Learning Predictions in here
	sig1 = tupdate.map(lambda d: (calc(d[1]))).reduce(lambda p,q:np.add(p,q))
	sig2 = tupdate.map(lambda d: (calc_s2(d[1]))).reduce(lambda p,q:np.add(p,q))

	weight = np.dot(np.linalg.inv(sig1),sig2)
	print "the weight vector is",weight
	print weight.shape

	ydash = tupdate.map(lambda d: np.dot(weight.T,d[1][:18])).collect()
	#--------------------- Learning Ends in here -------------------------#

	train  = tupdate.filter(lambda x: x[0][0] == 'en' and len(x[0][1])%2==0 )
	test = tupdate.filter(lambda x: x[0][0] == 'en' and len(x[0][1])%2!=0 ) 

	sig_t1 = train.map(lambda d: (calc(d[1]))).reduce(lambda p,q:np.add(p,q))
	sig_t2 = train.map(lambda d: (calc_s2(d[1]))).reduce(lambda p,q:np.add(p,q))
	train_weight = np.dot(np.linalg.inv(sig_t1),sig_t2)
	print "the weight vector is",train_weight
	print train_weight.shape

	learn_ydash = test.map(lambda d: np.dot(train_weight.T,d[1][:18]))

	#-----Code to calculate RMSE--------------------------#
	#update_yd needs to know the weight.T before hand
	tdash = tupdate.map(lambda d: update_yd(d))
	rmse =  math.sqrt(tdash.mean())
	print "the rmse error is ", rmse



if __name__ == '__main__':
	conf = SparkConf().setAppName('hw3').setMaster("master")
	sc = SparkContext(conf=conf)
	main()
