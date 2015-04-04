#!/bin/pyspark
import numpy as np
import time
from pyspark import SparkContext, SparkConf
import matplotlib.pyplot as plt

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

	output = np.linalg.lstsq(x, y)[0]
	print "the output for lstsq is ", output
	''''
	plt.plot(x, y, 'o', label='Original data', markersize=10)
	plt.plot(x, m*x + c, 'r', label='Fitted line')
	plt.legend()
	plt.show()
	'''

def main():
	tfile = readfile()
	dcollec = create_feature(tfile)
	print "the aggregated data in 6-22 collections is",dcollec
	dtarget = get_target_info()
	print "the data in 23rd file is", dtarget
	for key in dtarget:
		if dcollec.has_key(key):
			x = dcollec[key]
			x[18] = dtarget[key]
			dcollec[key] = x
		else:
			print "the key",key,"is not the dcollection... "
			time.sleep(2)
	print "the update aggregated data in dcollec is ",dcollec
	mat = np.matrix(dcollec.values())
	ols(mat)


if __name__ == '__main__':
	conf = SparkConf().setAppName('hw3').setMaster("master")
	sc = SparkContext(conf=conf)
	main()


'''
sc.textFile(base+'/'+str(fno)+'.txt')
tfile_rdd = tfile.map(lambda view: set_feature(view))
'''